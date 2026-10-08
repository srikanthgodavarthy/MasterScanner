import pandas as pd
from utils.rotate_candidates import drop_stale_snapshots, pick_first_not_falling


def _pool():
    return pd.DataFrame({
        "symbol": ["BEML", "AAA", "BBB"], "score": [90, 85, 80],
        "scan_date": ["2026-10-07", "2026-10-07", "2026-09-10"],
    })


def test_falling_top_candidate_is_skipped():
    pct = {"BEML": -5.0, "AAA": 0.4, "BBB": 1.0}
    row, p = pick_first_not_falling(_pool(), pct.get)
    assert row["symbol"] == "AAA" and p == 0.4


def test_all_falling_returns_none():
    row, p = pick_first_not_falling(_pool(), lambda s: -4.0)
    assert row is None and p is None


def test_unknown_live_change_fails_open():
    row, _ = pick_first_not_falling(_pool(), lambda s: None)
    assert row["symbol"] == "BEML"


def test_lookup_exception_fails_open():
    def boom(s): raise RuntimeError("x")
    row, _ = pick_first_not_falling(_pool(), boom)
    assert row["symbol"] == "BEML"


def test_threshold_boundary_and_check_cap():
    row, _ = pick_first_not_falling(_pool(), lambda s: -3.0)           # exactly -3 is skipped
    assert row is None
    row, _ = pick_first_not_falling(_pool(), lambda s: -4.0, max_checks=1)
    assert row is None


def test_stale_snapshot_rows_dropped():
    out = drop_stale_snapshots(_pool(), max_age_days=3)
    assert list(out["symbol"]) == ["BEML", "AAA"]
    # no scan_date column -> untouched
    assert len(drop_stale_snapshots(_pool().drop(columns="scan_date"))) == 3
