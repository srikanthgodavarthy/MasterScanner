"""News panel trims recaps of completed activity; keeps anything that can still move a trade."""
import json
import types
import warnings
import pytest

warnings.filterwarnings("ignore")
from utils.news_filter import recap_reason, split_recaps, is_classified_recap, collect_new_items


# ── layer 1: regex on the title ─────────────────────────────────────────────
@pytest.mark.parametrize("title", [
    "Adani Ent Share Price Live Updates: Adani Enterprises Records Minor Increase",
    "Asian Paints Share Price Live Updates: Asian Paints Daily Price Update",
    "Stock Market LIVE Updates: Sensex, Nifty trade flat",
    "Closing Bell: Sensex ends 300 points higher, Nifty above 22,700",
    "Sensex ends 412 points lower, Nifty settles below 22,600",
    "Zomato shares jump 5% in trade today",
    "Reliance hits 52-week high",
    "XYZ stock locked in upper circuit",
])
def test_recaps_are_hidden(title):
    assert recap_reason(title) is not None


@pytest.mark.parametrize("title", [
    # the other rows from the screenshot — none are completed-activity recaps
    "Which bank stock should you buy ahead of Q2 results - HDFC Bank, ICICI Bank, Yes Bank or Axis Bank?",
    "5 years, no returns: Are these 9 Nifty bluechip stocks finally ready for a turnaround?",
    "Bandhan Bank among 5 stocks showing White Marubozu pattern",
    # price move + a named catalyst -> carries news beyond the move
    "Tata Elxsi shares jump 7% after Q2 results, but Morgan Stanley, Motilal Oswal reiterate bearish calls",
    "Zomato shares jump 5% after bagging big order",
    "Reliance hits 52-week high on target price hike",
    # forward looking
    "Sensex may jump 500 points as global cues improve",
    "Infosys shares fall 3% ahead of Q2 results",
    "Stocks to watch today: Tata Motors, Infosys",
    "RBI policy preview: what to expect",
    "",
])
def test_news_that_can_still_move_a_trade_is_kept(title):
    assert recap_reason(title) is None


def test_split_recaps_annotates_reason_and_keeps_order():
    items = [{"title": "Asian Paints Share Price Live Updates: x"},
             {"title": "Zomato bags Rs 500 crore order"},
             {"title": "Closing Bell: Sensex ends 100 points higher"}]
    kept, hidden = split_recaps(items)
    assert [i["title"] for i in kept] == ["Zomato bags Rs 500 crore order"]
    assert [h["hidden_reason"] for h in hidden] == ["live / price-update page", "market wrap"]
    assert "hidden_reason" not in items[0]                       # caller's dicts untouched


# ── layer 2: classifier flag fails open ─────────────────────────────────────
@pytest.mark.parametrize("status,expected", [
    ("Recap", True), ("recap", True), (" RECAP ", True),
    ("New", False), (None, False), ("", False), ("Unclassified", False), (5, False),
])
def test_is_classified_recap_fails_open(status, expected):
    assert is_classified_recap({"status": status}) is expected


def _fake_client(results):
    msg = types.SimpleNamespace(content=json.dumps({"results": results}))
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
    create = lambda **kw: resp
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def test_classifier_parses_status_and_defaults_to_new(monkeypatch):
    pytest.importorskip("streamlit")
    import utils.news_sentiment as ns
    base = {"sentiment": "Positive", "event_type": "Earnings", "magnitude": "High",
            "horizon": "Days", "recommendation": "BUY", "note": "x"}
    monkeypatch.setattr(ns, "_get_news_model", lambda: "m")
    monkeypatch.setattr(ns, "get_client", lambda: _fake_client([
        {**base, "index": 0, "status": "Recap"},
        {**base, "index": 1, "status": "New"},
        {**base, "index": 2},                         # model omitted the field
        {**base, "index": 3, "status": "garbage"},    # model mangled it
    ]))
    out = ns._classify_batch(["a", "b", "c", "d"])
    assert [o["status"] for o in out] == ["Recap", "New", "New", "New"]


def test_unclassified_and_ratelimited_never_look_like_recaps(monkeypatch):
    pytest.importorskip("streamlit")
    import utils.news_sentiment as ns
    monkeypatch.setattr(ns, "get_client", lambda: None)
    assert not any(is_classified_recap(o) for o in ns._classify_batch(["a", "b"]))


# ── selection loop ──────────────────────────────────────────────────────────
def _pool(n):
    return [{"title": f"t{i}", "id": i} for i in range(n)]


def _tagger(recap_ids):
    return lambda batch: [{**it, "status": "Recap" if it["id"] in recap_ids else "New"} for it in batch]


def test_recaps_removed_and_reported():
    shown, hidden = collect_new_items(_pool(10), _tagger({1, 4}), cap=10, min_shown=6)
    assert [s["id"] for s in shown] == [0, 2, 3, 5, 6, 7, 8, 9]
    assert [h["id"] for h in hidden] == [1, 4] and hidden[0]["hidden_reason"]


def test_second_batch_only_when_too_few_survive():
    calls = []
    def tag(batch):
        calls.append(len(batch))
        return _tagger(set(range(0, 7)))(batch)       # first batch: ids 0-6 are recaps
    shown, hidden = collect_new_items(_pool(25), tag, cap=10, min_shown=6, max_rounds=2)
    assert calls == [10, 10]                           # 3 survivors < 6 -> classify next batch
    assert len(hidden) == 7
    assert len(shown) == 10 and [x['id'] for x in shown[:3]] == [7, 8, 9]   # 13 survivors, capped at 10

    calls.clear()
    collect_new_items(_pool(25), lambda b: (calls.append(1), _tagger(set())(b))[1], cap=10, min_shown=6)
    assert calls == [1]                                # plenty survive -> one Groq call only


def test_rounds_are_bounded_and_empty_pool_is_safe():
    calls = []
    def tag(batch):
        calls.append(1)
        return _tagger(set(range(100)))(batch)         # everything is a recap
    shown, hidden = collect_new_items(_pool(100), tag, cap=10, min_shown=6, max_rounds=2)
    assert shown == [] and len(hidden) == 20 and len(calls) == 2
    assert collect_new_items([], tag) == ([], [])
