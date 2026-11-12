"""tests/test_dore_winrate_diagnostic.py -- guards for scripts/dore_winrate_diagnostic.py"""
import importlib.util, pathlib
import numpy as np, pandas as pd, pytest

_p = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "dore_winrate_diagnostic.py"
_s = importlib.util.spec_from_file_location("dore_wr", _p); wr = importlib.util.module_from_spec(_s); _s.loader.exec_module(wr)


def _plans(n=60, seed=1):
    r = np.random.default_rng(seed)
    entry = r.uniform(5, 50, n)
    last = entry * r.choice([0.6, 0.65, 1.2, 1.5], n)
    return pd.DataFrame({
        "symbol": r.choice(["NIFTY", "TCS", "INFY"], n), "direction": "CE", "status": "CLOSED",
        "entry_locked": entry, "sl_locked": entry * 0.65, "target1_locked": entry * 1.5, "target2_locked": entry * 2.0,
        "last_premium": last, "mfe_premium": np.maximum(last, entry * 1.1), "mae_premium": np.minimum(last, entry * 0.9),
        "strike": 100.0, "entry_underlying": 97.0, "confidence_at_entry": r.uniform(68, 90, n),
        "gamma_blast_score_at_mint": r.uniform(30, 90, n),
        "entry_triggered_at": "2026-11-03T05:00:00Z", "closed_at": "2026-11-04T09:00:00Z",
        "expiry": "2026-11-24", "t1_hit_at": [("2026-11-03T08:00:00Z" if x >= 1.5 * e else "") for x, e in zip(last, entry)],
        "closed_reason_code": r.choice(["STOP_LOSS", "TIMEOUT", "TARGET_2"], n), "source": "", "activation_reason": "",
    })


def test_wilson_bounds_and_edges():
    lo, hi = wr.wilson(50, 100)
    assert lo < 50 < hi and 39 < lo < 41 and 59 < hi < 61
    assert wr.wilson(0, 0)[0] != wr.wilson(0, 0)[0]            # NaN, not a crash
    assert wr.wilson(10, 10)[1] == 100.0


def test_prepare_drops_never_entered_and_derives_geometry():
    df = _plans(); df.loc[0, "entry_locked"] = np.nan
    d = wr.prepare(df)
    assert len(d) == len(df) - 1
    assert d["sl_pct"].round(1).eq(-35.0).all() and d["t1_pct"].round(1).eq(50.0).all()
    assert d["breakeven_wr_pct"].round(1).eq(round(35 / 85 * 100, 1)).all()      # needs ~41% winners
    assert (d["win_pnl"] == (d["last_premium"] > d["entry_locked"])).all()


def test_slices_and_report_run_and_flag_small_cells():
    d = wr.add_buckets(wr.prepare(_plans()))
    t = wr.slice_table(d, "reason", min_n=1000)
    assert t["low_n"].all() and t["n"].sum() == len(d)
    txt = wr.report(wr.prepare(_plans()), min_n=8)
    for needle in ("break-even win rate", "how it closed", "blast score at mint", "low_n"):
        assert needle in txt


def test_empty_input_is_handled():
    assert "No closed" in wr.report(wr.prepare(_plans().iloc[0:0]))


def test_blast_promoted_slice_detects_activation_reason():
    df = _plans(); df.loc[:9, "activation_reason"] = "Gamma blast promotion (score 75 > 60, level BLAST)"
    d = wr.prepare(df)
    assert int(d["blast_promoted"].sum()) == 10
