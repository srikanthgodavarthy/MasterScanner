"""
Active Setups table (pages/scanner.py):
  * the FP source column is hidden (display only), and header/cells share one
    column list so they cannot drift;
  * the Volume cell always shows the volume ratio; a conflicted plan keeps its
    conflict as a hover tooltip instead of replacing the number.
"""
from pages import scanner as sc


def test_fp_column_hidden_and_cv4_kept():
    cols = sc._ap_source_columns()
    assert "FP" not in cols
    assert cols[-1] == "CV4"


def test_cells_match_header_column_count():
    cells = sc._ap_per_source_cells("LS", "", '{"LS": 100.0}', 100.0)
    assert cells.count("<td") == len(sc._ap_source_columns())


def test_fp_entry_is_not_rendered_even_for_fp_plans():
    cells = sc._ap_per_source_cells("FP", "", '{"FP": 123.45}', 123.45)
    assert "123.45" not in cells


def test_volume_cell_shows_ratio_for_conflicted_plan():
    cell = sc._ap_volume_cell(1.26, True, "bull/bear evidence")
    assert "1.3x" in cell and "Conflict</td>" not in cell
    assert 'title="Conflict: bull/bear evidence"' in cell


def test_volume_cell_plain_when_no_conflict():
    cell = sc._ap_volume_cell(0.3, False, "")
    assert cell == '<td class="col-num">0.3x</td>'


def test_volume_cell_handles_missing_and_nan():
    for bad in (None, 0, 0.0, float("nan"), "x"):
        assert ">—<" in sc._ap_volume_cell(bad, False, "")


def test_conflict_reason_is_html_escaped():
    cell = sc._ap_volume_cell(1.0, True, 'a"b<script>')
    assert "<script>" not in cell and 'a"b' not in cell
