"""
routes/stream.py
-----------------
§9 real-time update strategy: one /stream/prices SSE endpoint pushing a
JSON event per ticker update. static/js/live.js consumes this and
patches the specific <td>/arc element in place.

Open question (§9, carried into README): actual scan refresh cadence
(tick-by-tick vs. batch every N minutes) is not yet confirmed. Until
answered, this endpoint emits a small random-walk tick every 4s for
the top-of-book tickers as a stand-in — cheap to rip out once the
real cadence is known, since it's isolated to this one file.
"""
import json
import time
import random

from flask import Blueprint, Response, current_app

from services.scanner_service import get_scanner_rows

bp = Blueprint("stream", __name__)


def _price_events(app):
    with app.app_context():
        rows = get_scanner_rows()[:15]
    prices = {r.ticker: r.price for r in rows}
    convictions = {r.ticker: r.conviction_composite for r in rows}
    rng = random.Random(42)

    while True:
        ticker = rng.choice(list(prices.keys())) if prices else None
        if ticker:
            prices[ticker] = round(prices[ticker] * (1 + rng.uniform(-0.004, 0.004)), 2)
            convictions[ticker] = max(0, min(100, convictions[ticker] + rng.randint(-2, 2)))
            payload = {
                "ticker": ticker,
                "price": prices[ticker],
                "conviction": convictions[ticker],
            }
            yield f"event: tick\ndata: {json.dumps(payload)}\n\n"
        time.sleep(4)


@bp.get("/stream/prices")
def stream_prices():
    app = current_app._get_current_object()
    return Response(_price_events(app), mimetype="text/event-stream")
