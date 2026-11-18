"""tests/test_exit_engine_replay.py -- scripts/exit_engine_replay.py: point-in-time replay of the exit engine."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from utils.exit_intelligence_engine import ExitIntelligenceConfig

_p = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "exit_engine_replay.py"
_s = importlib.util.spec_from_file_location("exit_replay", _p)
er = importlib.util.module_from_spec(_s)
_s.loader.exec_module(er)
CFG = ExitIntelligenceConfig()


def _bars(close, seed=1, vol=2e6, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    close = np.asarray(close, float)
    op = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, len(close)))
    hi = np.maximum(op, close) * 1.004
    lo = np.minimum(op, close) * 0.996
    idx = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx)


def _rise_then_crash(n_up=170, n_dn=40):
    rng = np.random.default_rng(5)
    up = 100 * np.exp(0.0025 * np.arange(n_up)) * (1 + rng.normal(0, 0.006, n_up))
    dn = up[-1] * np.exp(-0.012 * np.arange(1, n_dn + 1)) * (1 + rng.normal(0, 0.006, n_dn))
    return _bars(np.r_[up, dn])


def _trade(df, entry_i, exit_i, stop_frac=0.85):
    entry = float(df["close"].iloc[entry_i])
    exit_px = float(df["close"].iloc[exit_i])
    return {"symbol": "AAA", "entry_date": df.index[entry_i], "entry_price": entry, "exit_date": df.index[exit_i],
            "exit_price": exit_px, "pnl_pct": (exit_px / entry - 1) * 100, "sl": entry * stop_frac, "exit_reason": "X"}


def test_engine_exits_a_crash_earlier_and_better_than_riding_it_down():
    df = _rise_then_crash()
    t = pd.DataFrame([_trade(df, 150, len(df) - 1, stop_frac=0.5)])             # wide stop: rode the crash
    res = er.evaluate_trades(t, lambda s: df, CFG)
    r = res.iloc[0]
    assert pd.notna(r["exit_signal_date"]) and r["exit_signal_date"] < t["exit_date"].iloc[0]
    assert r["exit_signal_pnl_pct"] > r["actual_pnl_pct"]                        # followed the engine: lost less
    s = er.summarize(res)
    assert s["net_vs_actual_pct"] > 0 and s["losers_flagged_pct"] == 100.0 and s["losers_avg_improvement_pct"] > 0


def test_quiet_period_leaves_the_trade_alone():
    df = _rise_then_crash()
    t = pd.DataFrame([_trade(df, 100, 125)])                                      # healthy stretch only
    res = er.evaluate_trades(t, lambda s: df, CFG)
    s = er.summarize(res)
    assert pd.isna(res.iloc[0]["exit_signal_date"]) and s["net_vs_actual_pct"] == 0.0


def test_no_lookahead_bars_after_the_trade_do_not_change_the_signal():
    df = _rise_then_crash()
    t = _trade(df, 150, 190, stop_frac=0.5)
    full = er.replay_trade(df, t["entry_date"], t["exit_date"], t["entry_price"], t["sl"], CFG)
    cut = er.replay_trade(df.iloc[: 191 + 1], t["entry_date"], t["exit_date"], t["entry_price"], t["sl"], CFG)
    assert (full["exit"] or {}).get("date") == (cut["exit"] or {}).get("date")
    assert (full["reduce"] or {}).get("date") == (cut["reduce"] or {}).get("date")


def test_fill_is_next_bar_open_never_the_signal_bar_close():
    df = _rise_then_crash()
    t = _trade(df, 150, len(df) - 1, stop_frac=0.5)
    hit = er.replay_trade(df, t["entry_date"], t["exit_date"], t["entry_price"], t["sl"], CFG)["exit"]
    assert hit is not None
    pos = df.index.get_loc(hit["date"])
    assert hit["price"] == pytest.approx(float(df["open"].iloc[pos]))
    assert df.index[pos - 1] < hit["date"]


def test_symbols_without_data_are_skipped_and_summary_handles_empty():
    df = _rise_then_crash()
    t = pd.DataFrame([_trade(df, 150, 190), {**_trade(df, 150, 190), "symbol": "NODATA"}])
    res = er.evaluate_trades(t, lambda s: df if s == "AAA" else None, CFG)
    assert list(res["symbol"]) == ["AAA"]
    assert er.summarize(pd.DataFrame()) == {"n": 0}


def test_confirm_closes_setting_changes_how_fast_a_break_exits():
    df = _rise_then_crash()
    t = pd.DataFrame([_trade(df, 150, len(df) - 1, stop_frac=0.5)])
    d1 = er.evaluate_trades(t, lambda s: df, ExitIntelligenceConfig(structure_break_confirm_closes=1)).iloc[0]["exit_signal_date"]
    d3 = er.evaluate_trades(t, lambda s: df, ExitIntelligenceConfig(structure_break_confirm_closes=3)).iloc[0]["exit_signal_date"]
    assert pd.notna(d1) and pd.notna(d3) and d1 <= d3


def test_cli_end_to_end_with_stubbed_prices(tmp_path, monkeypatch):
    df = _rise_then_crash()
    trades = pd.DataFrame([_trade(df, 150, len(df) - 1, stop_frac=0.5), _trade(df, 100, 125)])
    f = tmp_path / "trades.csv"
    trades.to_csv(f, index=False)
    monkeypatch.setattr(er, "yfinance_loader", lambda start, end, cache: (lambda sym: df))
    assert er.main(["--trades", str(f), "--confirm-closes", "1,2", "--out-dir", str(tmp_path), "--cache-dir", ""]) == 0
    summ = pd.read_csv(tmp_path / "exit_replay_summary.csv", index_col=0)
    assert list(summ.columns) == ["1", "2"] and "net_vs_actual_pct" in summ.index
    assert (tmp_path / "exit_replay_confirm1.csv").exists()
