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


# ── Entry column rename / entry time / Trigger column [2026-11-04] ───────────
def test_cv4_column_is_displayed_as_entry():
    assert sc._ap_source_header_label("CV4") == "Entry"
    assert sc._ap_source_header_label("PB") == "PB"


def test_entry_time_shown_beneath_price_on_entry_column_only():
    cells = sc._ap_per_source_cells("LS", "", '{"LS": 3301.42}', 3301.42, ("11:17:48", False))
    assert "3,301.42" in cells and "11:17:48" in cells
    assert cells.index("3,301.42") < cells.index("11:17:48")
    assert cells.count("<td") == len(sc._ap_source_columns())
    assert "~" not in cells


def test_entry_time_marked_approx_and_omitted_when_unknown():
    assert "~11:17:48" in sc._ap_per_source_cells("LS", "", "", 100.0, ("11:17:48", True))
    assert ":" not in sc._ap_per_source_cells("LS", "", "", 100.0, ("", False)).split("100.00")[1]


def test_entry_time_converts_utc_to_ist():
    # 05:47:48 UTC == 11:17:48 IST
    assert sc._ap_entry_time("2026-11-04T05:47:48+00:00", "2026-11-04T05:40:00+00:00") == ("11:17:48", False)
    assert sc._ap_entry_time("2026-11-04 05:47:48.5+00:00", "") == ("11:17:48", False)


def test_entry_time_legacy_date_only_falls_back_to_same_day_created_at():
    # old rows: activated_at was a date -> midnight UTC, no real time
    assert sc._ap_entry_time("2026-11-04T00:00:00+00:00", "2026-11-04T05:40:00+00:00") == ("11:10:00", True)
    # created on a different day -> no made-up time
    assert sc._ap_entry_time("2026-11-04T00:00:00+00:00", "2026-11-02T05:40:00+00:00") == ("", False)
    assert sc._ap_entry_time("", "") == ("", False)


def test_trigger_cell_pills_and_empty():
    cell = sc._ap_trigger_cell("Breakout, PB")
    assert "Breakout" in cell and "PB" in cell and cell.count("<span") == 2
    assert ">—<" in sc._ap_trigger_cell("")
    assert "opacity" in sc._ap_trigger_cell("Fib", frozen=False)


def test_trigger_cell_escapes_html():
    assert "<script>" not in sc._ap_trigger_cell("<script>")


def test_derive_trigger_label():
    from utils.setup_persistence import derive_trigger_label as d
    assert d({"Buy Type": "Fib+CCI"}) == "Fib+CCI"
    assert d({"Buy Type": "CmpBrk"}) == "Breakout"
    assert d({"Buy Type": "T1Pullback", "T1Path": "A"}) == "PB"
    assert d({"Buy Type": "Fib", "T1Path": "C", "FreshBase": True}) == "Fib, Breakout"
    assert d({"Buy Type": "-", "T1Path": "", "FreshBase": False}) == ""
    assert d({}) == "" and d(None) == ""
