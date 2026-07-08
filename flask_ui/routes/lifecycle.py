from flask import Blueprint, render_template, abort

from services.trade_service import (
    get_active_trades, get_closed_trades, exit_trade, trail_stop,
)

bp = Blueprint("lifecycle", __name__)


@bp.get("/trades")
def trades():
    return render_template(
        "lifecycle.html",
        active_trades=get_active_trades(),
        closed_trades=get_closed_trades(),
    )


@bp.post("/trades/<trade_id>/exit")
def post_exit(trade_id):
    """Journal-only exit (see services/trade_service.py docstring re:
    open question #4 — no broker API call is made)."""
    t = exit_trade(trade_id)
    if t is None:
        abort(404)
    # Move the row into the closed table via an out-of-band swap, and
    # remove it from the active table — see §8.4 decision in README.
    return render_template("partials/trade_exit_response.html", trade=t, oob="afterbegin:#closed-trades-table")


@bp.post("/trades/<trade_id>/trail-stop")
def post_trail_stop(trade_id):
    t = trail_stop(trade_id)
    if t is None:
        abort(404)
    return render_template("partials/trade_row.html", trade=t, closed=False)
