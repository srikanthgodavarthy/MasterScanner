"""
Active Plans "Expected Stock Move": show the stock TARGET in the plan's
direction (CE above spot, PE below) with a SIGNED percentage that matches it.

Plans minted before expected_price_at_mint existed used to fall back to a
direction-less "+/-INR42 (9.2%)". The price is recoverable from the stored
move (2 dp) and percentage-of-spot (3 dp), so those rows now show the target too.
"""
from __future__ import annotations

import re
import pandas as pd
import pytest

scanner = pytest.importorskip("pages.scanner")
view = scanner._expected_price_view

RUPEE = "\u20b9"


def _row(direction="CE", em=42.0, pct=9.188, price=None, **kw):
    r = {"direction": direction, "expected_move_at_mint": em, "expected_move_pct_at_mint": pct,
         "expected_price_at_mint": price}
    r.update(kw)
    return r


# ----------------------------------------------------------------- the view
def test_stored_price_is_used_as_is_and_pct_is_signed_by_direction():
    ce = view(_row("CE", 42.0, 9.188, price=498.60))
    assert ce["price"] == 498.60 and ce["spot"] == pytest.approx(456.60) and not ce["approx"]
    assert ce["pct"] == pytest.approx(9.198, abs=0.01) and ce["pct"] > 0
    pe = view(_row("PE", 42.0, 9.188, price=414.60))
    assert pe["price"] == 414.60 and pe["spot"] == pytest.approx(456.60) and pe["pct"] < 0 and not pe["up"]


def test_legacy_rows_get_the_price_reconstructed_in_the_plans_direction():
    spot = 457.10
    em = 42.0
    pct = round(em / spot * 100, 3)            # exactly how the mint stores it
    ce = view(_row("CE", em, pct))
    pe = view(_row("PE", em, pct))
    assert ce["approx"] and pe["approx"]
    assert ce["price"] == pytest.approx(spot + em, abs=0.25) and ce["price"] > ce["spot"]
    assert pe["price"] == pytest.approx(spot - em, abs=0.25) and pe["price"] < pe["spot"]
    assert ce["pct"] > 0 > pe["pct"]


@pytest.mark.parametrize("spot", [45.0, 180.0, 457.1, 1240.5, 2473.3, 4273.8, 9890.0])
@pytest.mark.parametrize("pct_move", [2.0, 5.5, 9.2, 12.9, 18.0])
@pytest.mark.parametrize("direction", ["CE", "PE"])
def test_reconstruction_is_accurate_across_prices_and_the_percentage_always_matches(spot, pct_move, direction):
    em_true = spot * pct_move / 100.0
    em, pct = round(em_true, 2), round(em_true / spot * 100, 3)    # as persisted
    v = view(_row(direction, em, pct))
    true_price = spot + em_true if direction == "CE" else spot - em_true
    assert abs(v["price"] - true_price) <= 0.35, (spot, pct_move, v)
    # the % shown is derived from the price shown, and agrees with the exact figure at 1 dp
    exact = (true_price - spot) / spot * 100
    assert abs(v["pct"] - exact) < 0.06
    assert (v["pct"] > 0) == (direction == "CE")
    assert abs(v["pct"] - (v["price"] - v["spot"]) / v["spot"] * 100) < 1e-9


@pytest.mark.parametrize("row", [
    _row("CE", None, 9.2), _row("CE", 0.0, 9.2), _row("CE", -5.0, 9.2),
    _row(None, 42.0, 9.2), _row("XX", 42.0, 9.2),
    _row("CE", 42.0, None), _row("CE", 42.0, 0.0), _row("CE", 42.0, -1.0),
    _row("CE", float("nan"), 9.2), _row("CE", 42.0, float("nan")),
])
def test_unrecoverable_rows_return_none_rather_than_a_wrong_price(row):
    assert view(row) is None


# ----------------------------------------- the real table, rows like the screenshot
def _table(rows):
    df = pd.DataFrame(rows)
    return scanner._dore_options_active_plans_table_html(df)


def _plan(symbol, direction, strike, em, pct, **kw):
    r = {"symbol": symbol, "direction": direction, "strike": strike, "expiry": "2026-11-23",
         "created_date": "2026-10-29", "status": "ACTIVE", "lifecycle_group": "ACTIVE",
         "confidence_at_entry": 70.0, "entry_locked": 2.0, "saved_stop_loss": 1.33,
         "saved_target1": 3.07, "saved_target2": 4.10, "last_premium": 2.5, "last_drift_pct": 25.0,
         "expected_move_at_mint": em, "expected_move_pct_at_mint": pct,
         "expected_move_source_at_mint": "ATR"}
    r.update(kw)
    return r


def test_legacy_ce_rows_show_a_target_above_spot_with_a_matching_signed_percent():
    html = _table([
        _plan("KOTAKBANK", "CE", 480, 42.0, round(42 / 456.5 * 100, 3)),
        _plan("TVSMOTOR", "CE", 4500, 500.0, round(500 / 4273.0 * 100, 3)),
        _plan("TCS", "CE", 2600, 319.0, round(319 / 2473.0 * 100, 3)),
    ])
    for em in ("42", "500", "319"):                                         # no direction-less band shown
        assert f"+/-{RUPEE}{em}" not in html and f"\u00b1{RUPEE}{em}" not in html
    assert html.count("\u25b2") >= 3 and "\u25bc" not in html                # all CE -> up
    assert f"+{RUPEE}42 (+9.2%)" in html
    assert f"+{RUPEE}500 (+11.7%)" in html
    assert f"+{RUPEE}319 (+12.9%)" in html
    assert re.search(rf"\u25b2 ~{RUPEE}498\.[0-9]{{2}}", html)               # ~ = reconstructed
    assert "reconstructed from the stored move and percentage" in html


def test_pe_rows_point_down_with_a_negative_percent_and_exact_prices_have_no_tilde():
    html = _table([_plan("INFY", "PE", 1500, 80.0, 5.0, expected_price_at_mint=1520.0)])
    assert "\u25bc" in html and "\u25b2" not in html
    assert f"\u2212{RUPEE}80 (\u22125.0%)" in html
    assert f"{RUPEE}1,520.00" in html and "~" not in html.split("INFY")[1].split("<td")[0] + html


def test_a_plan_with_no_usable_direction_still_renders_the_band_and_never_crashes():
    html = _table([_plan("ODDCO", None, 100, 10.0, 5.0)])
    assert f"\u00b1{RUPEE}10 (5.0%)" in html and "ODDCO" in html
