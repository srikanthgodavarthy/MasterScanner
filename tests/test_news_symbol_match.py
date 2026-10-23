"""News feed ticker chips must come from the headline, not the RSS summary."""
from utils.symbol_name_map import match_symbols
from utils.news_sentiment import enrich_symbols


def _tag(title, summary=""):
    return enrich_symbols([{"title": title, "summary": summary}])[0]["symbols"]


def test_summary_mentions_do_not_leak_into_chips():
    assert _tag("Kaynes Technology among 3 stocks showing White Marubozu pattern",
                "Other names: Bajaj Finance, Siemens") == []
    assert _tag("Swiggy among 4 F&O stocks with sharp rise in futures open interest",
                "Siemens, Hindalco see OI build-up") == []
    assert _tag("Gold price falls Rs 1,500/10g; silver plunges Rs 7,000/kg",
                "On MCX, gold futures slipped") == []


def test_headline_company_still_tagged():
    assert _tag("Infosys shares jump 3% ahead of Q2 earnings announcement") == ["INFY"]
    assert _tag("Kotak Mahindra Bank Q2 results preview: PAT may rise") == ["KOTAKBANK"]
    assert _tag("HPCL shares in focus as Q1 net profit slips 6% YoY") == ["HINDPETRO"]


def test_mcx_commodity_context_not_the_stock():
    assert "MCX" not in match_symbols("MCX gold slips as rate hike bets rise")
    assert "MCX" in match_symbols("MCX shares surge after Q2 profit beat")


def test_global_feed_relevance_filter():
    from utils.news_feed import is_india_relevant
    assert is_india_relevant("Nifty ends higher as banks rally")
    assert is_india_relevant("Infosys wins $1.5 billion deal")
    assert is_india_relevant("Asian stocks rise", "Foreign investors return to India")
    assert not is_india_relevant("Wall Street falls as Fed signals higher rates")
    assert not is_india_relevant("Bitcoin tops $100,000")
