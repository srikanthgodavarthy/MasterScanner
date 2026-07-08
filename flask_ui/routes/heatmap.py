from flask import Blueprint, render_template

from services.scanner_service import get_sector_heatmap

bp = Blueprint("heatmap", __name__)


@bp.get("/sectors")
def sectors():
    tiles = get_sector_heatmap()
    return render_template("heatmap.html", tiles=tiles)
