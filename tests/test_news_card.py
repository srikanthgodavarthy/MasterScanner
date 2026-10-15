"""Card-style news feed: headline first, one action slot, dated times, escaped feed text."""
import re
import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

warnings.filterwarnings("ignore")
from utils.news_card import action_view, dedupe_items, title_key, when_label

UTC = timezone.utc
NOW = datetime(2026, 10, 14, 5, 30, tzinfo=UTC)          # 11:00 AM IST on 14 Oct 2026


def ist(y, mo, d, h, mi):                                # an IST wall-clock time as a UTC datetime
    return (datetime(y, mo, d, h, mi) - timedelta(hours=5, minutes=30)).replace(tzinfo=UTC)


# ── when_label ──────────────────────────────────────────────────────────────
def test_when_label_today_yesterday_older_and_missing():
    assert when_label(ist(2026, 10, 14, 9, 13), NOW) == "09:13 AM"
    assert when_label(ist(2026, 10, 13, 16, 27), NOW) == "Yesterday 04:27 PM"
    assert when_label(ist(2026, 10, 9, 21, 23), NOW) == "9 Oct 09:23 PM"
    assert when_label(None, NOW) == "—"


def test_when_label_uses_ist_day_boundary_not_utc():
    # 00:30 IST on 14 Oct is 19:00 UTC on 13 Oct — still "today" in IST
    assert when_label(datetime(2026, 10, 13, 19, 0, tzinfo=UTC), NOW) == "12:30 AM"
    assert when_label(datetime(2026, 10, 14, 5, 0), NOW) == "10:30 AM"        # naive treated as UTC


# ── action_view ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("rec,conf,label,kind,sub", [
    ("BUY", "High", "BUY", "buy", ""), ("AVOID", "Medium", "AVOID", "avoid", ""),
    ("HOLD", "High", "No signal", "none", ""), (None, "High", "No signal", "none", ""),
    ("BUY", "Low", "No signal", "none", "low confidence"),
    ("AVOID", None, "No signal", "none", "low confidence"),
    ("buy", "high", "BUY", "buy", ""),
])
def test_action_view(rec, conf, label, kind, sub):
    v = action_view(rec, conf)
    assert (v["label"], v["kind"], v["sub"]) == (label, kind, sub)


def test_action_view_keeps_raw_read_for_tooltip():
    assert action_view("AVOID", "Low")["raw"] == "AVOID" and action_view("HOLD", "High")["raw"] == "HOLD"


# ── dedupe ──────────────────────────────────────────────────────────────────
def test_duplicate_headlines_collapse_keeping_first():
    a = {"title": "Tata Consumer Q4 net profit falls 19% to Rs 217 cr, firm declares dividend of Rs 7.75 per share", "id": 1}
    b = {**a, "id": 2, "title": a["title"].upper() + " "}
    c = {"title": "Tata Elxsi Q4 net profit, revenue decline QoQ", "id": 3}
    assert [i["id"] for i in dedupe_items([a, c, b])] == [1, 3]
    assert len(dedupe_items([{"title": ""}, {"title": ""}])) == 2            # untitled never merged
    assert title_key("Hello,  World!") == title_key("hello world")


# ── rendering ───────────────────────────────────────────────────────────────
def _item(**kw):
    base = dict(title="Headline", link="https://x.test/a", source="ET Markets", published=ist(2026, 10, 14, 9, 13),
                symbols=["LICI"], sector="Financials", sentiment="Positive", magnitude="Medium",
                event_type="Corporate Action", recommendation="HOLD", horizon=None, impact_note="note")
    base.update(kw)
    return base


@pytest.fixture
def cards(monkeypatch):
    pytest.importorskip("streamlit")
    import pages.dashboard as d
    import utils.news_card as nc
    monkeypatch.setattr(nc, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: NOW)}))
    return lambda items, scan=None: d._news_impact_cards_html(items, scan if scan is not None else pd.DataFrame())


def test_cards_not_grid_and_headline_comes_first(cards):
    html = cards([_item(title="LIC Housing Finance among 4 F&O stocks"), _item(title="Second story")])
    assert html.count('class="nf-card') == 2 and "ni-grid" not in html and "<table" not in html
    first = html.split('class="nf-card')[1]
    assert first.index("nf-headline") < first.index("nf-meta") < first.index("nf-impact")


def test_action_slot_only_speaks_for_real_signals(cards):
    html = cards([_item(recommendation="BUY", magnitude="High", sentiment="Positive"),
                  _item(recommendation="HOLD"),
                  _item(recommendation="AVOID", magnitude="Low", sentiment="Negative")])
    a, b, c = html.split('class="nf-card')[1:]
    assert 'nf-act-pill nf-buy">BUY' in a and "No signal" not in a
    assert "No signal" in b and "nf-act-pill" not in b
    assert "No signal" in c and "low confidence" in c and "nf-act-pill" not in c
    assert "Model read: AVOID" in c                       # nothing the model said is lost


def test_impact_and_confidence_merge_into_one_badge(cards):
    html = cards([_item(sentiment="Negative", magnitude="High")])
    assert 'nf-impact nf-neg">Negative · High' in html and "nf-neg" in html.split('class="nf-card')[1].split(">")[0]


def test_tickers_wrap_in_one_row_with_fresh_pct(cards):
    scan = pd.DataFrame({"Stock": ["ICICIBANK", "HDFCBANK"], "%Chg": [-0.8, float("nan")]})
    html = cards([_item(symbols=["ICICIBANK", "HDFCBANK", "AXISBANK"])], scan)
    assert html.count('class="nf-chip"') == 3
    assert "-0.80%" in html                                # badge from the single %Chg source
    assert html.count("%</span>") == 1                     # NaN / unknown -> no badge (never "nan%")
    assert not re.search(r"\bnan\b|nan%", html.lower())      # "financials" must not trip this


def test_dates_and_market_wide_label(cards):
    html = cards([_item(published=ist(2026, 10, 13, 16, 27), symbols=[], sector=None)])
    assert "Yesterday 04:27 PM" in html and "Market-wide" in html and 'class="nf-chip"' not in html


def test_feed_text_is_escaped(cards):
    evil = '<script>alert(1)</script> "quoted" & more'
    html = cards([_item(title=evil, source="<b>src</b>", link='https://x.test/"><img src=x>',
                        impact_note='" onmouseover="x', event_type="<i>x</i>")])
    assert "<script>" not in html and "<img" not in html and "<b>src" not in html and "<i>x" not in html
    assert "&lt;script&gt;" in html and 'onmouseover="x' not in html
