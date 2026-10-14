"""
utils/news_filter.py — drop news that only reports something already done.

The News Impact panel is for headlines that can still move a trade. A
headline that merely reports a completed activity — a live price-update
page, a market wrap, "shares jump 7%" — tells you what the price already did;
by the time it is read the move is in the price, and the classifier then
dresses it up as BUY / HOLD / AVOID anyway.

Two layers (this module is layer 1, free and deterministic):
  1. recap_reason(title): regex over the TITLE only. Runs BEFORE any Groq
     call, so recaps cost no tokens and never take one of the capped slots.
  2. The classifier's own `status` field ("New" | "Recap", see
     utils/news_sentiment.py) catches the recaps regex can't see.

Bias: precision over recall. A false drop hides a real catalyst; a missed
recap just shows one extra row that the classifier may still catch. So:
  * anything forward-looking ("may", "likely", "ahead of", "preview") is kept;
  * a price-move headline that ALSO names a catalyst (order win, upgrade,
    target price, probe, ...) is kept.
Pure Python, no streamlit — safe to import anywhere and to unit test.
"""
from __future__ import annotations

import re
from typing import Iterable

_I = re.IGNORECASE

# Live-update / auto-generated price pages: "X Share Price Live Updates",
# "Stock price today", "Daily Price Update", live blogs.
_LIVE_PAGE = re.compile(
    r"\b(share|stock)\s+price\s+(live\s+updates?|today|live)\b"
    r"|\blive\s+updates?\b"
    r"|\blive\s*blog\b"
    r"|\bdaily\s+price\s+update\b"
    r"|\bstock\s+market\s+(today|live)\b"
    r"|\bmarket\s+live\b"
    r"|^\s*live\s*[:\-]",
    _I,
)

# Index / market wraps: "Closing Bell", "Sensex ends 300 points higher".
_MARKET_WRAP = re.compile(
    r"\bclosing\s+bell\b|\bmarket\s+wrap\b|\bmarket\s+close\b"
    r"|\b(sensex|nifty|bank\s*nifty)\b[^.]{0,60}\b(ends?|closes?|settles?|snaps?|extends?|erases?|slips?|tanks?|"
    r"surges?|jumps?|rall(?:y|ies)|gains?|falls?|drops?)\b[^.]{0,40}(\bpoints?\b|\bpts\b|%)",
    _I,
)

# A completed move on a single name: "shares jump 7%", "stock falls 3%".
_PRICE_MOVE = re.compile(
    r"\b(shares?|stocks?)\b[^.]{0,60}\b(jump|surge|soar|rall(?:y|ies)|rise|rises|gain|climb|zoom|spike|slump|tumble|"
    r"plunge|fall|drop|sink|crash|slide|slip|skid|tank)s?\b[^.]{0,40}\d+(?:\.\d+)?\s?%",
    _I,
)
# "hits 52-week high", "locked in upper circuit" — a level already reached.
_LEVEL_HIT = re.compile(
    r"\b(hits?|touch(?:es)?|scales?|zooms?\s+to)\s+(?:a\s+)?(?:fresh\s+)?(?:52[- ]week|all[- ]time|record)\s+(?:high|low)\b"
    r"|\b(upper|lower)\s+circuit\b",
    _I,
)

# Forward-looking language: never treat these as recaps.
_FORWARD = re.compile(
    r"\b(may|might|could|likely|expected?|set\s+to|poised|ahead\s+of|preview|what\s+to\s+expect|"
    r"should\s+you|to\s+(?:open|rise|fall|hit|launch|raise|buy|sell|acquire)|outlook|eyes?|targets?)\b",
    _I,
)

# A named catalyst alongside a price move means the headline carries news
# beyond the move itself — keep it and let the classifier judge.
_CATALYST = re.compile(
    r"\b(orders?|contracts?|bags?|wins?|secures?|acquir\w*|acquisition|merger|stake|buyback|dividend|bonus|split|"
    r"target\s+price|upgrades?|downgrades?|initiates?|reiterates?|bullish|bearish|guidance|approval|approves?|"
    r"probe|penalty|fined?|ban|raid|sebi|rbi|ipo|fund\s*raise|fundrais\w*|launch\w*)\b",
    _I,
)


def recap_reason(title: str) -> "str | None":
    """Why this headline is a recap of something already done, or None.

    Layer 1 of the recap filter — title only, regex only, no cost."""
    t = (title or "").strip()
    if not t:
        return None
    if _LIVE_PAGE.search(t):
        return "live / price-update page"
    if _FORWARD.search(t):
        return None
    if _MARKET_WRAP.search(t):
        return "market wrap"
    if (_PRICE_MOVE.search(t) or _LEVEL_HIT.search(t)) and not _CATALYST.search(t):
        return "completed price move"
    return None


def split_recaps(items: Iterable[dict]) -> "tuple[list[dict], list[dict]]":
    """(kept, hidden). Hidden items carry `hidden_reason` so the UI can list
    what was trimmed and why instead of dropping rows silently."""
    kept: list[dict] = []
    hidden: list[dict] = []
    for it in items:
        reason = recap_reason(str(it.get("title", "")))
        if reason:
            hidden.append({**it, "hidden_reason": reason})
        else:
            kept.append(it)
    return kept, hidden


def is_classified_recap(item: dict) -> bool:
    """Layer 2: the classifier flagged this as a recap. Fails OPEN — a missing
    or unknown status (no API key, rate limit, model omission) is never a recap."""
    return str(item.get("status") or "").strip().title() == "Recap"


def collect_new_items(pool: "list[dict]", tag_fn, cap: int = 10, min_shown: int = 6,
                      max_rounds: int = 2) -> "tuple[list[dict], list[dict]]":
    """Classify `pool` in batches of `cap` via `tag_fn` and drop classifier-flagged
    recaps. Dropping rows leaves the panel short, so when fewer than `min_shown`
    survive, classify the next batch — bounded by `max_rounds` so one refresh
    can't spend unbounded Groq calls. Returns (shown[:cap], hidden)."""
    shown: list[dict] = []
    hidden: list[dict] = []
    cursor = 0
    for _ in range(max_rounds):
        batch = pool[cursor:cursor + cap]
        cursor += len(batch)
        if not batch:
            break
        for it in tag_fn(batch):
            if is_classified_recap(it):
                hidden.append({**it, "hidden_reason": "classified as recap (already done)"})
            else:
                shown.append(it)
        if len(shown) >= min_shown:
            break
    return shown[:cap], hidden
