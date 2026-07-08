from flask import Blueprint, render_template, request

from services.scanner_service import get_scanner_rows
from services.conviction_service import get_conviction_breakdown, get_conviction_timeline, get_blockers

bp = Blueprint("dashboard", __name__)


@bp.get("/")
def index():
    setup = request.args.get("setup", "all")
    rows = get_scanner_rows(setup=setup)
    opportunities = [r for r in rows if r.category == "Elite Opportunity"][:6]
    return render_template(
        "dashboard.html",
        rows=rows,
        opportunities=opportunities,
        active_setup=setup,
    )


@bp.get("/partials/scanner-rows")
def scanner_rows_partial():
    setup = request.args.get("setup", "all")
    rows = get_scanner_rows(setup=setup)
    return render_template("partials/scanner_rows_table.html", rows=rows, active_setup=setup)


@bp.get("/partials/opportunities")
def opportunities_partial():
    rows = get_scanner_rows()
    opportunities = [r for r in rows if r.category == "Elite Opportunity"][:6]
    return render_template("partials/opportunities.html", opportunities=opportunities)


@bp.get("/partials/analysis-drawer/<ticker>")
def analysis_drawer(ticker):
    breakdown = get_conviction_breakdown(ticker)
    timeline = get_conviction_timeline(ticker)
    blockers = get_blockers(ticker)
    return render_template(
        "partials/analysis_drawer.html",
        ticker=ticker,
        breakdown=breakdown,
        timeline=timeline,
        blockers=blockers,
    )
