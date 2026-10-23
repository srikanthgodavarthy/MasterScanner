"""
utils/news_card.py — pure helpers for the card-style News Impact feed.

The old feed was a 9-column grid. Row height was set by the ticker chips, the
headline sat after five narrow metadata columns, Impact / Recommendation / the
coloured border all said the same thing (Positive -> BUY, Neutral -> HOLD,
Negative -> AVOID), and times carried no date, so yesterday evening's
results sat among this morning's headlines unlabelled. The card layout (see
pages/dashboard.py::_news_impact_cards_html) fixes that; the logic that has
no business in a render function lives here so it can be unit tested.

No streamlit import.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from utils.time_utils import IST

_STRONG_CONFIDENCE = ("Medium", "High")


def when_label(published: Optional[datetime], now: Optional[datetime] = None) -> str:
    """'09:13 AM' for today (IST), 'Yesterday 04:27 PM', else '13 Oct 04:27 PM'.
    '—' when there is no timestamp. `published` is UTC (utils/news_feed.py)."""
    if not published:
        return "—"
    try:
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        p = published.astimezone(IST)
        n = (now or datetime.now(timezone.utc))
        if n.tzinfo is None:
            n = n.replace(tzinfo=timezone.utc)
        today = n.astimezone(IST).date()
        clock = p.strftime("%I:%M %p")
        if p.date() == today:
            return clock
        if p.date() == today - timedelta(days=1):
            return f"Yesterday {clock}"
        return f"{p.day} {p.strftime('%b')} {clock}"
    except Exception:
        return "—"


def action_view(recommendation: Optional[str], confidence: Optional[str]) -> dict:
    """What to show in the card's action slot.

    Only a BUY / AVOID the model is at least Medium-confident about is a signal.
    HOLD is the absence of one, and a Low-confidence BUY/AVOID is not strong
    enough to dress up as an action — both read "No signal". The raw
    recommendation stays available as `raw` for the tooltip.
    Returns {"label", "kind": "buy"|"avoid"|"none", "sub", "raw"}."""
    rec = (recommendation or "").strip().upper() or None
    conf = (confidence or "").strip().title() or None
    if rec in ("BUY", "AVOID") and conf in _STRONG_CONFIDENCE:
        return {"label": rec, "kind": rec.lower(), "sub": "", "raw": rec}
    if rec in ("BUY", "AVOID"):
        return {"label": "No signal", "kind": "none", "sub": "low confidence", "raw": rec}
    return {"label": "No signal", "kind": "none", "sub": "", "raw": rec}


_NORM = re.compile(r"[^a-z0-9]+")


def title_key(title: str) -> str:
    """Case / punctuation / whitespace-insensitive key for duplicate headlines."""
    return _NORM.sub(" ", (title or "").lower()).strip()


def dedupe_items(items: Iterable[dict]) -> "list[dict]":
    """Drop repeats of the same headline (the same story arrives via several
    feeds, or twice from one), keeping the FIRST occurrence so the input's
    order — newest first — is preserved. Untitled items are kept."""
    seen: set = set()
    out: list = []
    for it in items:
        k = title_key(str(it.get("title", "")))
        if k:
            if k in seen:
                continue
            seen.add(k)
        out.append(it)
    return out


_MAG_RANK = {"High": 0, "Medium": 1, "Low": 2}


def split_by_sentiment(items: Iterable[dict]) -> "tuple[list[dict], list[dict], list[dict]]":
    """(positive, negative, other) for the two-column News Impact view.

    `other` is everything that is not a clear Positive/Negative read (Neutral,
    Unclassified, RateLimited, missing). Within each of positive/negative the
    strongest reads come first (High > Medium > Low); the sort is stable, so
    equal-magnitude items keep the input's newest-first order."""
    pos: list = []
    neg: list = []
    other: list = []
    for it in items:
        s = it.get("sentiment")
        (pos if s == "Positive" else neg if s == "Negative" else other).append(it)
    rank = lambda it: _MAG_RANK.get(it.get("magnitude"), 3)
    pos.sort(key=rank)
    neg.sort(key=rank)
    return pos, neg, other
