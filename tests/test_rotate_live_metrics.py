"""ROTATE targets must come from TODAY's scan, not a weeks-old lifecycle snapshot."""
import datetime as dt
import warnings
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
pytest.importorskip("streamlit")

from utils.rotate_candidates import build_live_metrics, drop_stale_snapshots, ROTATE_MAX_AGE_VS_TODAY_DAYS

TODAY = dt.date(2026, 10, 14)


def _snap(rows):
    return pd.DataFrame(rows)


def _row(sym, rec, lead, conv, eq, score, chg=None, bar=None):
    r = {"Stock": sym, "Recommendation": rec, "CV1_Leadership": lead, "CV1_Conviction": conv,
         "CV1_EntryQuality": eq, "Score": score, "Lifecycle": "FORMING"}
    if chg is not None:
        r["%Chg"] = chg
    if bar:
        r["BarDate"] = bar
    return r


def _lifecycle(sym, cat, date, score):
    return {"symbol": sym, "scan_date": date, "category": cat, "stage": "FORMING", "leadership": 70,
            "conviction": 66, "entry_quality": 57, "score": score, "rs_composite": 50.0, "trend_quality": 60}


def test_live_category_overrides_old_lifecycle_watch():
    snap = _snap([_row("BEML", "Skip", 54, 46, 57, 40)])
    lc = pd.DataFrame([_lifecycle("BEML", "Watch", "2026-09-21", 80),
                       _lifecycle("OLDONLY", "Watch", "2026-09-21", 70)])
    m = build_live_metrics(snap, lc, TODAY)
    beml = m[m.symbol == "BEML"]
    assert len(beml) == 1 and beml.iloc[0]["category"] == "Skip" and beml.iloc[0]["source"] == "live"
    old = m[m.symbol == "OLDONLY"].iloc[0]                       # fallback keeps its own old date
    assert old["source"] == "lifecycle" and old["scan_date"] == "2026-09-21"


def test_no_snapshot_date_means_no_live_rows():
    m = build_live_metrics(_snap([_row("X", "Watch", 60, 55, 55, 70)]), pd.DataFrame(), None)
    assert m.empty


def test_pct_chg_today_is_freshness_checked(monkeypatch):
    import utils.pct_change as pc
    monkeypatch.setattr(pc, "_market_open", lambda: True)
    monkeypatch.setattr(pc, "_today_str", lambda: "2026-10-14")
    snap = _snap([_row("FRESH", "Watch", 60, 55, 55, 70, chg=-1.2, bar="2026-10-14"),
                  _row("STALE", "Watch", 60, 55, 55, 70, chg=4.3, bar="2026-10-13")])
    m = build_live_metrics(snap, None, TODAY).set_index("symbol")
    assert m.loc["FRESH", "pct_chg_today"] == -1.2 and pd.isna(m.loc["STALE", "pct_chg_today"])


def test_freshness_gate_measured_against_today_not_newest_row():
    pool = pd.DataFrame({"symbol": ["A", "B", "C"],
                         "scan_date": ["2026-09-21", "2026-10-11", "2026-10-14"]})
    # legacy mode: relative to newest row -> only the 3 day window around 10-14
    assert list(drop_stale_snapshots(pool).symbol) == ["B", "C"]
    only_old = pool[pool.symbol == "A"]
    assert len(drop_stale_snapshots(only_old)) == 1            # the blind spot: its own newest
    assert drop_stale_snapshots(only_old, ROTATE_MAX_AGE_VS_TODAY_DAYS, reference_date=TODAY).empty
    bad = pd.DataFrame({"symbol": ["Z"], "scan_date": ["garbage"]})
    assert drop_stale_snapshots(bad, 4, reference_date=TODAY).empty   # unparseable -> fail closed


# ── _best_swap end to end ────────────────────────────────────────────────
@pytest.fixture
def pf(monkeypatch):
    import pages.portfolio as pf
    monkeypatch.setattr(pf, "_today_ist", lambda: TODAY)
    monkeypatch.setattr(pf, "_candidate_today_pct", lambda s: None)
    return pf


def _metrics(rows):
    return pd.DataFrame(rows)


def test_skip_today_blocks_target_even_if_old_row_was_watch(pf):
    lm = build_live_metrics(
        _snap([_row("BEML", "Skip", 54, 46, 57, 90), _row("GOOD", "Watch", 62, 55, 55, 70)]),
        pd.DataFrame([_lifecycle("BEML", "Watch", "2026-09-21", 95)]), TODAY)
    best = pf._best_swap({"lm_score": 40}, lm, held_symbols=set(), excluded_symbols=set())
    assert best is not None and best["symbol"] == "GOOD"


def test_only_stale_watch_rows_gives_no_target(pf):
    lm = _metrics([{"symbol": s, "scan_date": "2026-09-21", "category": "Watch", "score": 80}
                   for s in ("BEML", "TEGA", "PFOCUS")])
    assert pf._best_swap({"lm_score": 40}, lm, set(), set()) is None


def test_missing_category_column_fails_closed(pf):
    lm = _metrics([{"symbol": "X", "scan_date": "2026-10-14", "score": 99}])
    assert pf._best_swap({"lm_score": 40}, lm, set(), set()) is None


def test_snapshot_pct_used_before_network(pf, monkeypatch):
    calls = []
    monkeypatch.setattr(pf, "_candidate_today_pct", lambda s: calls.append(s))
    lm = build_live_metrics(
        _snap([_row("DOWN", "Watch", 62, 55, 55, 90, chg=-5.0), _row("OK", "Watch", 62, 55, 55, 70, chg=0.5)]),
        None, TODAY)
    best = pf._best_swap({"lm_score": 40}, lm, set(), set())
    assert best["symbol"] == "OK" and calls == []              # fell back to nothing: snapshot answered both
