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
array of "title — summary" strings. The model tags each result with the
0-based index of the input item it belongs to (see _classify_batch) --
matched back by identity rather than trusted by list position, since a
lighter model occasionally drops or reorders one item out of a batch.
This keeps call count low against Groq's free-tier rate limits while
staying well inside context limits for even the longest realistic batch.

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

from utils.groq_client import get_client, _is_available
from utils.sector_map import get_sector
from utils.symbol_name_map import match_symbols

logger = logging.getLogger(__name__)

_BATCH_SIZE = 15

# 2026-07-19: news classification gets its OWN model, deliberately
# decoupled from utils.groq_client.get_model() / DEFAULT_MODEL.
# Context: this panel hit a 429 (tokens-per-day cap on
# llama-3.3-70b-versatile) mid-session -- the free-tier Groq org has a
# single shared daily token budget across every caller of that model, and
# a news panel polling 5 RSS feeds with a 5-field classification per
# headline is a heavy, recurring consumer of that budget all on its own.
# Rather than fight other Groq usage for the same 100k TPD pool, news
# classification runs on a smaller/cheaper model with its own headroom.
# Override via GROQ_NEWS_MODEL in secrets if you want to point this at
# something else (or back at 70b) without touching groq_client.py's
# shared default.
_NEWS_MODEL = "llama-3.1-8b-instant"

# Summaries are stored at up to 280 chars (utils/news_feed.py) for
# display, but the classifier doesn't need that much to judge direction/
# category -- trimmed further here purely to cut prompt tokens per call.
_SUMMARY_CHARS_FOR_LLM = 150

_EVENT_TYPES = (
    "Earnings", "Order/Contract", "Regulatory/Legal", "Rating/Brokerage",
    "Promoter/Insider", "M&A", "Macro/Policy", "Corporate Action", "Other",
)
_MAGNITUDES = ("High", "Medium", "Low")
_HORIZONS = ("Immediate", "Days", "Weeks")
# 2026-07-19: Recommendation is now a direct Groq read of the headline
# itself (BUY/HOLD/AVOID), deliberately independent of any scan state --
# see pages/scanner.py _resolve_scan_recommendation, which is now labeled
# "Current State" and shows the scan's own tier (Elite/Execute/etc.) as a
# SEPARATE column. Previously this panel's "Recommendation" column
# reused the scan tier directly, which conflated "what does this
# headline suggest" with "what does the scanner currently say about this
# stock" -- two different questions that happened to share a column.
_RECOMMENDATIONS = ("BUY", "HOLD", "AVOID")

_SYSTEM_PROMPT = f"""You are a terse financial news classifier for Indian equity markets.
For each headline (with a short summary, when available) you are given,
classify its likely near-term market impact.

Return ONLY a JSON object of the form:
{{"results": [{{"index": <0-based position of this item in the input array>,
  "sentiment": "Positive" | "Negative" | "Neutral",
  "event_type": "Earnings" | "Order/Contract" | "Regulatory/Legal" | "Rating/Brokerage" | "Promoter/Insider" | "M&A" | "Macro/Policy" | "Corporate Action" | "Other",
  "magnitude": "High" | "Medium" | "Low",
  "horizon": "Immediate" | "Days" | "Weeks",
  "recommendation": "BUY" | "HOLD" | "AVOID",
  "note": "<=15 words"}}, ...]}}

Rules:
- Exactly one result per input item. "index" MUST equal that item's
  0-based position in the input array -- this is how results are matched
  back, not the order you list them in.
- Every index from 0 to (input length - 1) must appear exactly once.
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
- "recommendation": your own terse trading read of THIS headline alone,
  independent of any external system state. BUY = the news is a clear
  bullish trigger worth acting on. AVOID = a clear bearish trigger, stay
  away / exit. HOLD = informational, mixed, low-magnitude, or not
  actionable on its own.
- "note": a terse reason (max 15 words), not a restatement of the headline.
- No markdown, no commentary outside the JSON object."""


def _get_news_model() -> str:
    """GROQ_NEWS_MODEL override via secrets, falling back to _NEWS_MODEL.
    Deliberately separate from groq_client.get_model() -- see the module
    docstring / _NEWS_MODEL comment for why."""
    try:
        return st.secrets.get("GROQ_NEWS_MODEL", _NEWS_MODEL) or _NEWS_MODEL
    except Exception:
        return _NEWS_MODEL


def _classify_batch(texts: list[str]) -> list[dict]:
    """One Groq call for one batch of 'title — summary' strings. Returns a
    fail-soft list of same length as input, so callers never have to
    special-case failures. Two distinct failure shapes on purpose:
      - 'Unclassified': no client (key missing) or an unexpected error.
      - 'RateLimited': the daily/per-minute token cap was hit -- this is
        a *known, temporary* state, not a broken integration, so it gets
        its own label rather than being lumped in with 'Unclassified'
        (which previously made "no key configured" and "hit the daily
        cap" look identical in the UI -- no way to tell them apart)."""
    fallback = [
        {"sentiment": "Unclassified", "event_type": None, "magnitude": None,
         "horizon": None, "recommendation": None, "note": ""}
        for _ in texts
    ]

    client = get_client()
    if client is None:
        return fallback

    try:
        resp = client.chat.completions.create(
            model=_get_news_model(),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(texts)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1800,
        )
        payload = json.loads(resp.choices[0].message.content)
        raw_results = payload.get("results", [])

        # 2026-07-19 fix: match results back to inputs by the model's
        # "index" field, not list position. A lighter model (this runs on
        # llama-3.1-8b-instant, see _NEWS_MODEL) occasionally drops or
        # merges one item out of a batch -- if we trusted position, every
        # item AFTER the dropped one would silently shift and come back
        # attributed to the wrong headline (confidently wrong, not just
        # missing). Index-matching means a dropped item only affects
        # itself; everything else stays correctly attributed.
        by_index: dict[int, dict] = {}
        for r in raw_results:
            idx = r.get("index")
            if not isinstance(idx, int) or not (0 <= idx < len(texts)):
                continue
            if idx in by_index:
                continue  # duplicate index from the model; keep the first

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

            recommendation = str(r.get("recommendation", "")).strip().upper()
            if recommendation not in _RECOMMENDATIONS:
                # sentiment-based fallback if the model omitted/mangled
                # this field but still gave a usable sentiment read
                recommendation = {
                    "Positive": "BUY", "Negative": "AVOID", "Neutral": "HOLD",
                }.get(sentiment)

            by_index[idx] = {
                "sentiment": sentiment,
                "event_type": event_type,
                "magnitude": magnitude,
                "horizon": horizon,
                "recommendation": recommendation,
                "note": str(r.get("note", "")).strip()[:140],
            }

        missing = [i for i in range(len(texts)) if i not in by_index]
        if missing:
            logger.warning(
                "news_sentiment: model omitted/misindexed %d/%d items (indices %s); "
                "backfilling those as Unclassified -- everything else is unaffected.",
                len(missing), len(texts), missing,
            )

        return [by_index.get(i, fallback[i]) for i in range(len(texts))]

    except Exception as exc:
        exc_text = str(exc)
        is_rate_limit = (
            "429" in exc_text
            or "rate_limit" in exc_text.lower()
            or getattr(exc, "status_code", None) == 429
        )
        if is_rate_limit:
            logger.warning("news_sentiment: Groq rate-limited: %s", exc)
            return [
                {"sentiment": "RateLimited", "event_type": None, "magnitude": None,
                 "horizon": None, "recommendation": None,
                 "note": "Groq token limit reached — retry later"}
                for _ in texts
            ]
        logger.warning("news_sentiment: Groq classification failed: %s", exc)
        return fallback


@st.cache_data(ttl=1800, show_spinner=False)
def _classify_cached(pairs: tuple[tuple[str, str], ...]) -> list[dict]:
    """Cache wrapper -- st.cache_data needs hashable args, hence tuples of
    (title, summary) rather than dicts."""
    texts = [
        f"{title} — {summary[:_SUMMARY_CHARS_FOR_LLM]}".strip(" —") if summary else title
        for title, summary in pairs
    ]
    out: list[dict] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i:i + _BATCH_SIZE]
        out.extend(_classify_batch(batch))
    return out


def enrich_symbols(items: list[dict]) -> list[dict]:
    """Deterministic, regex-based symbol/sector attachment -- NO LLM call,
    NO Groq token cost. Split out from tag_news() specifically so callers
    can prioritize/filter items by "does this even mention a stock"
    BEFORE paying for classification, rather than classifying everything
    fetched and only capping what's rendered afterward (see
    pages/scanner.py _news_impact_panel -- 2026-07-19: this was the
    actual biggest source of wasted Groq spend, not the schema/model
    choice: every deduped headline from all 5 feeds was being classified
    even though only ~10 are ever shown)."""
    enriched = []
    for item in items:
        symbols = match_symbols(f"{item['title']} {item.get('summary', '')}")
        sector = get_sector(symbols[0]) if symbols else None
        enriched.append({**item, "symbols": symbols, "sector": sector})
    return enriched


def tag_news(items: list[dict]) -> list[dict]:
    """
    Enrich raw feed items (from utils.news_feed.fetch_all_news) with:
      - symbols:        list[str]           (regex-matched NSE tickers, may be empty)
      - sector:         str | None          (sector of the first matched symbol, if any)
      - sentiment:      'Positive' | 'Negative' | 'Neutral' | 'Unclassified' | 'RateLimited'
      - event_type:     str | None          (Earnings, Order/Contract, ... or None)
      - magnitude:      'High' | 'Medium' | 'Low' | None
      - horizon:        'Immediate' | 'Days' | 'Weeks' | None
      - recommendation: 'BUY' | 'HOLD' | 'AVOID' | None  (Groq's own read of
                         THIS headline alone -- deliberately independent of
                         any scan state; see pages/scanner.py "Current
                         State" column for the scan-tier-based view)
      - impact_note:    str

    Classification is skipped (all items 'Unclassified', other fields None)
    if GROQ_API_KEY isn't configured -- the panel still renders with raw
    headlines + symbol tags, it just won't have the bullish/bearish read.

    Callers that want to filter/prioritize BEFORE spending Groq tokens
    should call enrich_symbols() themselves first (it's idempotent and
    cheap to call again here) rather than passing the full unfiltered
    feed straight into this function -- see pages/scanner.py.

    NOTE: this is a display-only enrichment. It intentionally does not
    feed CV1, the Promotion Engine, or the Decision Orchestrator -- it's
    context for a human reading the News Impact panel, not a scored
    input. If/when you want it wired into scoring, that's a deliberate
    follow-on change, not something to bolt on here silently.
    """
    if not items:
        return []

    enriched = enrich_symbols(items)

    if not _is_available():
        for item in enriched:
            item["sentiment"] = "Unclassified"
            item["event_type"] = None
            item["magnitude"] = None
            item["horizon"] = None
            item["recommendation"] = None
            item["impact_note"] = "Groq API key not configured"
        return enriched

    pairs = tuple((item["title"], item.get("summary", "")) for item in enriched)
    classifications = _classify_cached(pairs)

    for item, cls in zip(enriched, classifications):
        item["sentiment"] = cls["sentiment"]
        item["event_type"] = cls["event_type"]
        item["magnitude"] = cls["magnitude"]
        item["horizon"] = cls["horizon"]
        item["recommendation"] = cls["recommendation"]
        item["impact_note"] = cls["note"]

    return enriched
