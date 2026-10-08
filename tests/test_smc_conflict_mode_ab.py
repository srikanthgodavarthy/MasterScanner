"""Verdict logic of scripts/smc_conflict_mode_ab.py on synthetic trade frames."""
import importlib.util, pathlib
import numpy as np
import pandas as pd

_p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "smc_conflict_mode_ab.py"
_spec = importlib.util.spec_from_file_location("smc_ab", _p)
ab = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ab)


def _frame(seed, n_sym=40, per=12, shift=0.0, conflict=0.8):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sym):
        for k in range(per):
            rows.append(dict(symbol=f"S{s}", entry_date=pd.Timestamp("2025-01-01") + pd.Timedelta(days=k * 20 + s),
                             pnl_pct=rng.normal(shift, 3.0), r_multiple=rng.normal(shift / 5, 1.0),
                             smc_state_label="CONFLICT" if rng.random() < conflict else "NEUTRAL"))
    return pd.DataFrame(rows)


def test_identical_arms_are_inconclusive():
    f = _frame(1)
    r = ab.analyse(f, f.copy())
    assert "inconclusive" in r["verdict"] and abs(r["diff_mean_pnl"]["point"]) < 1e-9


def test_clearly_better_arm_is_flagged_better():
    L = _frame(1, shift=0.0); C = L.copy(); C["pnl_pct"] = C["pnl_pct"] + 2.5
    assert "BETTER" in ab.analyse(L, C)["verdict"]


def test_clearly_worse_arm_is_flagged_worse():
    L = _frame(1, shift=0.0); C = L.copy(); C["pnl_pct"] = C["pnl_pct"] - 2.5
    assert "WORSE" in ab.analyse(L, C)["verdict"]


def test_added_and_dropped_trades_are_keyed_on_symbol_and_date():
    L = _frame(1); C = L.iloc[10:].copy()                 # flip drops the first 10 trades
    extra = _frame(2, n_sym=2, per=3); extra["symbol"] = "NEW"
    C = pd.concat([C, extra], ignore_index=True)
    r = ab.analyse(L, C)
    assert r["dropped_by_flip"]["n"] == 10 and r["added_by_flip"]["n"] == 6


def test_shadow_compares_admitted_subset_only():
    L = _frame(1); L["passed_gate"] = True
    C = L.copy(); C["passed_gate"] = False
    assert ab.analyse(L, C, shadow=True)["metrics"]["confirmed_break"]["n"] == 0
