"""Active Plans rows for symbols missing from the scan: no 'nan%', no fake zeros."""
import warnings
import pytest

warnings.filterwarnings("ignore")
pytest.importorskip("streamlit")
import pages.scanner as sc


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "x"])
def test_pnl_cell_never_renders_nan(bad):
    html = sc._ap_pnl_cell(bad)
    assert "nan" not in html.lower() and "—" in html


def test_pnl_cell_values():
    assert "+4.9%" in sc._ap_pnl_cell(4.9) and "-3.9%" in sc._ap_pnl_cell(-3.86)
    assert "0.0%" in sc._ap_pnl_cell(0.0)


def test_score_cells_dash_when_missing():
    assert sc._ap_num(None, ".1f") == "—" and sc._ap_num(float("nan"), ".0f") == "—"
    assert sc._ap_num(58.7, ".1f") == "58.7" and sc._ap_num(0.0, ".0f") == "0"


def test_quote_uses_scan_row_when_it_has_a_price(monkeypatch):
    monkeypatch.setattr(sc, "_plan_fallback_quote", lambda s: pytest.fail("fallback not needed"))
    assert sc._resolve_plan_quote({"cmp": 4410.0, "pct_chg": 0.77}, "CPPLUS") == (4410.0, 0.77)


def test_quote_falls_back_for_symbol_missing_from_scan(monkeypatch):
    monkeypatch.setattr(sc, "_plan_fallback_quote", lambda s: {"cmp": 1315.75, "pct_chg": -0.4})
    assert sc._resolve_plan_quote({}, "PVRINOX") == (1315.75, -0.4)
    # in scan but no usable price (NaN Entry): same fallback, scan %Chg kept if it has one
    assert sc._resolve_plan_quote({"cmp": float("nan"), "pct_chg": 1.1}, "X") == (1315.75, 1.1)


def test_quote_unavailable_everywhere_gives_zero_and_none(monkeypatch):
    monkeypatch.setattr(sc, "_plan_fallback_quote", lambda s: {})
    assert sc._resolve_plan_quote({}, "GONE") == (0.0, None)
