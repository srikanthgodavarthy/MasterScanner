"""
Active Plans table (2026-10-28, SG request): Confidence Score, Expected Stock
Move, Theta, Gamma and the Greeks scenario favourable move are shown; Source
and Status are removed.

The three Greeks-derived numbers and the expected move are frozen at mint
(observation only). Theta is never persisted when the chain contradicted the
configured theta basis.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import re
import warnings

import pandas as pd
import pytest

warnings.filterwarnings("ignore")
pytest.importorskip("streamlit")

import pages.scanner as sc
import utils.supabase_client as sb
from utils.dore_live_state import _TechPlanView
from utils.dore_options_persistence import (
    DoreOptionsPlan, DoreOptionsPlanStatus, _greeks_at_mint,
    active_plan_rows, enrich_trade_plans_with_persistence,
)

_spec = importlib.util.spec_from_file_location(
    "_phase4_helpers", pathlib.Path(__file__).parent / "test_phase4_cv4_dore_integration.py")
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)

GREEKS_OK = {"ok": True, "theta": -12.4, "gamma": 0.00042, "fav_pct": 38.6, "method": "taylor",
             "reliable": True, "theta_verified": True}

NEW_COLS = ("expected_move_at_mint", "expected_move_pct_at_mint", "greeks_theta_at_mint",
            "greeks_gamma_at_mint", "greeks_fav_pct_at_mint", "greeks_method_at_mint",
            "greeks_reliable_at_mint", "greeks_theta_verified_at_mint")
# Third optional-column group (own migration): the stock price the expected move points to.
PRICE_COLS = ("expected_price_at_mint",)
ALL_AT_MINT_KEYS = NEW_COLS + PRICE_COLS


# ── _greeks_at_mint ──────────────────────────────────────────────────────────
def test_greeks_at_mint_happy_path():
    o = _greeks_at_mint({"expected_move": -118.0, "current_price": 24000.0, "greeks": dict(GREEKS_OK)})
    assert o["expected_move_at_mint"] == 118.0                      # magnitude; direction lives on the plan
    assert o["expected_move_pct_at_mint"] == pytest.approx(0.492, abs=1e-3)
    assert o["greeks_theta_at_mint"] == -12.4 and o["greeks_gamma_at_mint"] == 0.00042
    assert o["greeks_fav_pct_at_mint"] == 38.6 and o["greeks_method_at_mint"] == "taylor"
    assert o["greeks_reliable_at_mint"] is True and o["greeks_theta_verified_at_mint"] is True


def test_theta_is_not_persisted_when_basis_contradicted():
    g = {**GREEKS_OK, "theta_verified": False, "reliable": False, "method": "bs", "fav_pct": 21.0}
    o = _greeks_at_mint({"expected_move": 5.0, "current_price": 100.0, "greeks": g})
    assert o["greeks_theta_at_mint"] is None                         # the mis-scaled number is dropped
    assert o["greeks_theta_verified_at_mint"] is False
    assert o["greeks_fav_pct_at_mint"] == 21.0 and o["greeks_reliable_at_mint"] is False


@pytest.mark.parametrize("row", [
    {}, None, "x", {"greeks": None}, {"greeks": {"ok": False, "why": "no greeks"}},
    {"expected_move": float("nan"), "greeks": {"ok": True, "theta": float("nan"), "gamma": None}},
])
def test_greeks_at_mint_never_raises_and_never_fabricates(row):
    o = _greeks_at_mint(row)
    assert set(o) == set(ALL_AT_MINT_KEYS)
    assert o["greeks_theta_at_mint"] is None and o["greeks_fav_pct_at_mint"] is None
    assert o["expected_move_at_mint"] is None
    assert not any(isinstance(v, float) and math.isnan(v) for v in o.values())


def test_expected_move_pct_absent_without_spot():
    o = _greeks_at_mint({"expected_move": 50.0, "current_price": None})
    assert o["expected_move_at_mint"] == 50.0 and o["expected_move_pct_at_mint"] is None


# ── mint -> db dict -> reload -> table row ────────────────────────────────────
def _plan_dict(**over):
    close = _h._strong_uptrend_close()
    price = close[-1]
    plan = _h._run({**_h._base_row(price, bullish=True), **_h.CV4_COLS_BULLISH},
                   close, _h._option_data_for(price))
    d = plan.to_dict()
    d.update({"expected_move": 3.2, "current_price": price, "greeks": dict(GREEKS_OK)})
    d.update(over)
    return d


def _mint(pdict):
    view = _TechPlanView(plan=pdict, live={"current_premium": pdict["primary"]["premium"],
                                           "live_underlying_price": pdict["current_price"],
                                           "premium_change_pct": None})
    _, updated = enrich_trade_plans_with_persistence([view], existing_plans={})
    return updated[0]


def test_mint_freezes_expected_move_and_greeks_and_round_trips():
    plan = _mint(_plan_dict())
    assert plan.greeks_fav_pct_at_mint == 38.6 and plan.greeks_gamma_at_mint == 0.00042
    assert plan.expected_move_at_mint == 3.2 and plan.expected_move_pct_at_mint is not None
    db = plan.to_db_dict()
    assert all(c in db for c in NEW_COLS)
    back = sb._dore_options_plan_from_row(db)
    for c in NEW_COLS:
        assert getattr(back, c) == getattr(plan, c), c
    row = active_plan_rows({plan.contract_key: back})[0]
    assert all(c in row for c in NEW_COLS) and row["greeks_fav_pct_at_mint"] == 38.6


def test_old_rows_without_the_new_columns_still_load():
    db = _mint(_plan_dict()).to_db_dict()
    for c in NEW_COLS:
        db.pop(c)
    old = sb._dore_options_plan_from_row(db)
    assert old.greeks_fav_pct_at_mint is None and old.greeks_method_at_mint == ""
    row = active_plan_rows({old.contract_key: old})[0]
    assert row["greeks_theta_at_mint"] is None


# ── DB: migration + deploy-before-migration safety ───────────────────────────
def test_migration_declares_every_new_column_idempotently_and_ddl_matches():
    for c in NEW_COLS:
        assert c in sb.DORE_OPTIONS_PLANS_GREEKS_MIGRATION_SQL
        assert c in sb.SCHEMA_SQL
    assert sb.DORE_OPTIONS_PLANS_GREEKS_MIGRATION_SQL.count("ADD COLUMN IF NOT EXISTS") == len(NEW_COLS)
    assert tuple(sb._DORE_GREEKS_OPTIONAL_COLUMNS) == NEW_COLS


class _FakeDB:
    def __init__(self, missing=()):
        self.missing, self.calls = set(missing), []

    def is_available(self):
        return True

    def upsert_rows(self, table, rows, conflict_cols):
        self.calls.append(rows)
        for r in rows:
            for c in r:
                if c in self.missing:       # Postgres names ONE column per error
                    raise RuntimeError(f'column "{c}" of relation "{table}" does not exist')


def test_upsert_survives_greeks_migration_not_run_and_keeps_blast_data(monkeypatch):
    fake = _FakeDB(missing=NEW_COLS)
    monkeypatch.setattr(sb, "db", fake)
    row = _mint(_plan_dict(gamma_blast={"level": "WATCH", "score": 58.0, "risk_flags": []})).to_db_dict()
    assert sb.upsert_dore_options_plans_batch([row]) is True
    assert len(fake.calls) == 2
    sent = fake.calls[1][0]
    assert not any(c in sent for c in NEW_COLS)                       # whole Greeks group dropped
    assert sent["gamma_blast_level_at_mint"] == "WATCH"               # other group untouched
    assert sent["plan_id"] == row["plan_id"]


def test_upsert_drops_both_groups_when_neither_migration_ran(monkeypatch):
    fake = _FakeDB(missing=NEW_COLS + tuple(sb._DORE_OPTIONAL_COLUMNS))
    monkeypatch.setattr(sb, "db", fake)
    assert sb.upsert_dore_options_plans_batch([_mint(_plan_dict()).to_db_dict()]) is True
    final = fake.calls[-1][0]
    assert not any(c in final for c in NEW_COLS + tuple(sb._DORE_OPTIONAL_COLUMNS))


# ── the table ────────────────────────────────────────────────────────────────
def _df(**over):
    base = {
        "symbol": "DIVISLAB", "direction": "CE", "strike": 9900.0, "expiry": "2026-09-29",
        "confidence_at_entry": 69.0, "entry_locked": 19.5, "last_premium": 21.0, "last_drift_pct": 7.7,
        "saved_stop_loss": 14.47, "saved_target1": 25.73, "saved_target2": 30.15,
        "ce_iv_at_mint": 19.4, "pe_iv_at_mint": 21.0, "pcr_at_mint": 1.2,
        "expected_move_at_mint": 142.0, "expected_move_pct_at_mint": 1.45,
        "greeks_theta_at_mint": -4.3, "greeks_gamma_at_mint": 0.00231, "greeks_fav_pct_at_mint": 41.7,
        "greeks_method_at_mint": "taylor", "greeks_reliable_at_mint": True, "greeks_theta_verified_at_mint": True,
        "last_seen_at": "2026-09-24T09:40:00+00:00", "plan_age_days": 2, "plan_status_label": "🟢 Active",
        "is_inactive": False, "source": "LS", "status": "ACTIVE",
    }
    base.update(over)
    return pd.DataFrame([base])


def _headers(html_):
    return re.findall(r"<th[^>]*>([^<]*)</th>", html_)


def test_columns_are_exactly_the_requested_set_in_order_without_source_or_status():
    h = _headers(sc._dore_options_active_plans_table_html(_df()))
    assert h == ["Symbol", "Direction", "Strike", "Confidence Score", "Strike PCR", "Expected Stock Move",
                 "IV", "Theta", "Gamma", "Blast", "Last Known Premium",
                 "Entry (Locked)", "Stop Loss", "Target 1", "Target 2", "P&L", "Last Seen",
                 "Days Active", "Expiry"]
    assert "Source" not in h and "Status" not in h
    assert "Greeks Fav. Move" not in h                                 # removed: redundant with the targets
    assert h.index("Confidence Score") < h.index("Strike PCR")        # swapped


def test_every_row_has_one_cell_per_header():
    out = sc._dore_options_active_plans_table_html(_df())
    body = out.split("<tbody>")[1]
    assert body.count("<td") == len(_headers(out))


def test_cells_render_the_new_values():
    out = sc._dore_options_active_plans_table_html(_df())
    assert "69" in out and "🔵" in out                                # confidence, 55-74 band
    assert "±₹142 (1.5%)" in out or "±₹142 (1.4%)" in out            # a +/- band: rupees + % of spot
    assert "-4.30" in out                                              # theta ₹/day
    assert "0.00231" in out                                            # gamma
    assert "+42%" not in out and "Fav" not in out                      # scenario column is gone
    assert ">LS<" not in out and "PB</span>" not in out                # no Source badge


def test_expected_move_falls_back_to_a_band_when_the_price_was_not_recorded():
    """Plans minted before expected_price_at_mint existed: no exact level to show."""
    for d in ("CE", "PE"):
        out = sc._dore_options_active_plans_table_html(_df(direction=d))
        assert "±₹142" in out and "predates the exact price" in out


def test_expected_move_shows_the_definite_price_in_the_plans_direction():
    ce = sc._dore_options_active_plans_table_html(_df(direction="CE", expected_price_at_mint=10042.0))
    assert "▲ ₹10,042.00" in ce and ("+₹142 (1.5%)" in ce or "+₹142 (1.4%)" in ce) and "#3fb950" in ce
    assert "spot at mint ₹9,900.00 + ₹142" in ce and "opposite side of the band is ₹9,758.00" in ce
    pe = sc._dore_options_active_plans_table_html(_df(direction="PE", expected_price_at_mint=9758.0))
    assert "▼ ₹9,758.00" in pe and ("−₹142 (1.5%)" in pe or "−₹142 (1.4%)" in pe) and "spot at mint ₹9,900.00 − ₹142" in pe
    assert "opposite side of the band is ₹10,042.00" in pe


def test_entry_time_sits_under_the_locked_price_in_ist_without_a_date():
    out = sc._dore_options_active_plans_table_html(_df(entry_triggered_at="2026-10-28T04:39:00+00:00"))
    assert "₹19.50<div" in out and ">10:09</div>" in out               # time sits directly under the price; 04:39 UTC = 10:09 IST
    assert "2026-10-28T" not in out.split("<tbody>")[1]
    bare = sc._dore_options_active_plans_table_html(_df())
    assert 'title="Entry time (IST)"' not in bare                      # no stamp -> no empty sub-line
    naive = sc._dore_options_active_plans_table_html(_df(entry_triggered_at="2026-10-28 04:39:00"))
    assert ">10:09</div>" in naive                                     # naive stamps are read as UTC


def test_price_is_persisted_at_mint_above_spot_for_ce_and_below_for_pe():
    ce = _greeks_at_mint({"expected_move": 319.0, "current_price": 2454.0, "direction": "CE"})
    pe = _greeks_at_mint({"expected_move": 319.0, "current_price": 2454.0, "direction": "PE"})
    assert ce["expected_price_at_mint"] == 2773.0 and pe["expected_price_at_mint"] == 2135.0
    assert _greeks_at_mint({"expected_move": 319.0, "current_price": 2454.0})["expected_price_at_mint"] is None
    assert _greeks_at_mint({"expected_move": 319.0, "direction": "CE"})["expected_price_at_mint"] is None


def test_theta_withheld_and_flagged_when_basis_unverified():
    out = sc._dore_options_active_plans_table_html(_df(
        greeks_theta_at_mint=None, greeks_theta_verified_at_mint=False,
        greeks_reliable_at_mint=False, greeks_fav_pct_at_mint=21.0, greeks_method_at_mint="bs"))
    assert "theta_verified = False" in out and "⚠" in out


def test_missing_greeks_render_dashes_not_nan_or_zero():
    out = sc._dore_options_active_plans_table_html(_df(
        confidence_at_entry=0.0, expected_move_at_mint=None, expected_move_pct_at_mint=None,
        greeks_theta_at_mint=None, greeks_gamma_at_mint=None, greeks_fav_pct_at_mint=None,
        greeks_method_at_mint="", greeks_reliable_at_mint=None, greeks_theta_verified_at_mint=None))
    assert "nan" not in out.lower() and "None" not in out
    assert "0.00000" not in out and "+0%" not in out


def test_nan_inputs_are_treated_as_missing():
    nan = float("nan")
    out = sc._dore_options_active_plans_table_html(_df(
        expected_move_at_mint=nan, greeks_theta_at_mint=nan, greeks_gamma_at_mint=nan,
        greeks_fav_pct_at_mint=nan, confidence_at_entry=nan))
    assert "nan" not in out.lower()


def test_inactive_plan_is_dimmed_and_keeps_its_status_text_as_a_tooltip():
    out = sc._dore_options_active_plans_table_html(_df(
        is_inactive=True, plan_status_label="⚪ Inactive · expired (9d)"))
    assert 'opacity:.55' in out and "Inactive · expired" in out
    assert "<th>Status</th>" not in out
    live = sc._dore_options_active_plans_table_html(_df())
    assert "opacity:.55" not in live


def test_empty_df_message_unchanged():
    assert "No open plans" in sc._dore_options_active_plans_table_html(pd.DataFrame())
