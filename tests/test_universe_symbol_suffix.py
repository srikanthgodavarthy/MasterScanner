"""No ".NS.NS": harness universe symbols are bare; backtest tickers idempotent."""
import importlib.util, pathlib, sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_ab():
    spec = importlib.util.spec_from_file_location("smc_conflict_mode_ab", ROOT / "scripts" / "smc_conflict_mode_ab.py")
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m


def test_normaliser_returns_bare_symbols():
    ab = _load_ab()
    assert ab._normalise_nse_symbol(" acc ") == "ACC"
    assert ab._normalise_nse_symbol("ACC.NS") == "ACC"
    assert ab._normalise_nse_symbol("") == ""


def test_synthetic_symbols_are_bare():
    ab = _load_ab()
    data, _ = ab._synthetic_data(3)
    assert all(not k.endswith(".NS") for k in data)


def test_fetch_bt_batch_never_double_suffixes(monkeypatch):
    import utils.backtest_engine as be
    seen = {}
    def fake_dl(tickers, **kw):
        seen["t"] = list(tickers); return pd.DataFrame()
    monkeypatch.setattr(be, "yf_download_with_retry", fake_dl)
    be._fetch_bt_batch.clear() if hasattr(be._fetch_bt_batch, "clear") else None
    be._fetch_bt_batch(("ACC", "ABB.NS", "abb.ns"))
    assert seen["t"] == ["ACC.NS", "ABB.NS", "abb.ns"]
    assert not any(t.upper().endswith(".NS.NS") for t in seen["t"])
