from flask import Blueprint, render_template, request, abort

from services.scanner_service import get_stock_row, get_ohlcv_series
from services.conviction_service import get_conviction_breakdown, get_conviction_timeline, get_blockers

bp = Blueprint("stock_detail", __name__)


@bp.get("/stock/<ticker>")
def stock_detail(ticker):
    row = get_stock_row(ticker)
    if row is None:
        abort(404)
    series = get_ohlcv_series(ticker, "3M")
    breakdown = get_conviction_breakdown(ticker)
    timeline = get_conviction_timeline(ticker)
    blockers = get_blockers(ticker)
    return render_template(
        "stock_detail.html",
        row=row,
        series=series,
        breakdown=breakdown,
        timeline=timeline,
        blockers=blockers,
        timeframe="3M",
    )


@bp.get("/partials/chart/<ticker>")
def chart_partial(ticker):
    timeframe = request.args.get("tf", "3M")
    row = get_stock_row(ticker)
    if row is None:
        abort(404)
    series = get_ohlcv_series(ticker, timeframe)
    return render_template("partials/chart_svg.html", row=row, series=series, timeframe=timeframe)
