from flask import Blueprint, render_template, request

from services.scanner_service import get_scanner_rows

bp = Blueprint("mobile", __name__)


@bp.get("/m")
def mobile_home():
    setup = request.args.get("setup", "all")
    rows = get_scanner_rows(setup=setup)
    return render_template("mobile.html", rows=rows, active_setup=setup)
