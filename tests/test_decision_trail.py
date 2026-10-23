"""
Decision trail (P1 review 2.1 / 2.3).

build_decision_trail() and the fields it adds to score_stock rows are ADDITIVE and DIAGNOSTIC.
These tests pin three things:
  1. the pure function: every stage result and every stable reason code
  2. score_stock integration: fields present, internally consistent, and tier/action behaviour
     identical to the pre-change baseline (tests/fixtures/decision_trail_baseline.json)
  3. a failure inside tier assembly is no longer invisible
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import random

import numpy as np
import pytest

from tests import test_cv4_smc_toggles as T
from utils import scanner_engine as se
from utils.scanner_engine import DECISION_REASON_CODES, RECOMMENDATION_RANK, build_decision_trail

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "decision_trail_baseline.json")
R = RECOMMENDATION_RANK


def _kw(**over):
    base = dict(
        cv4_base_tier="Actionable", de_gate_on=False, de_gate_reason="", natural_class="WATCH",
        base_rank=R["Actionable"], natural_rank=0, promo_rank=0,
        promo_applicable=False, promo_promoted=False, promo_bypassed=False, promo_tier=None,
        pre_gate_tier="Actionable", final_tier="Actionable",
        smc_gate_enabled=True, smc_gate_would_be_tier="Actionable",
        action_raw="✅ BUY", action_final="✅ BUY",
    )
    base.update(over)
    return base


# ──────────────────────────────────────────────────────────────────
#  1. pure function
# ──────────────────────────────────────────────────────────────────
class TestPureFunction:
    def test_clean_pass_has_no_codes(self):
        out = build_decision_trail(**_kw())
        assert out["Decision_Reason_Codes"] == ""
        assert out["Decision_Trail"] == ("CV4=Actionable;DE_GATE=OFF;NATURAL=WATCH;PROMOTION=NOT_APPLICABLE;"
                                         "COMPOSE=Actionable@BASE;SMC_GATE=UNCHANGED;ACTION=UNCHANGED")
        assert out["SMC_Gate_Changed_Tier"] is False and out["Final_Tier_Source"] == "BASE"

    def test_bypass_promotion_then_smc_downgrade_then_action_cap(self):
        out = build_decision_trail(**_kw(
            cv4_base_tier="Skip", base_rank=R["Skip"], natural_class="SKIP",
            promo_rank=R["Execute"], promo_applicable=True, promo_promoted=True, promo_bypassed=True,
            promo_tier="Execute", pre_gate_tier="Execute", final_tier="Watch",
            smc_gate_would_be_tier="Watch", action_raw="✅ BUY", action_final="👁 WATCH"))
        assert out["Decision_Reason_Codes"] == "CV4_BELOW_ACTIONABLE|SMC_GATE_DOWNGRADED|ACTION_CAPPED_BY_TIER"
        assert "PROMOTION=BYPASS_PROMOTED:Execute" in out["Decision_Trail"]
        assert "COMPOSE=Execute@PROMOTION" in out["Decision_Trail"]
        assert "SMC_GATE=DOWNGRADED:Watch" in out["Decision_Trail"]
        assert out["Pre_Gate_Tier"] == "Execute" and out["SMC_Gate_Changed_Tier"] is True
        assert out["Final_Tier_Source"] == "PROMOTION"

    @pytest.mark.parametrize("reason,code", [("hard_stop", "DE_GATE_HARD_STOP"),
                                             ("lifecycle=AVOID", "DE_GATE_LIFECYCLE_AVOID"),
                                             ("lifecycle=EXTENDED", "DE_GATE_LIFECYCLE_EXTENDED")])
    def test_decision_engine_gate_codes(self, reason, code):
        out = build_decision_trail(**_kw(cv4_base_tier="Actionable", de_gate_on=True, de_gate_reason=reason,
                                         base_rank=R["Watch"], pre_gate_tier="Watch", final_tier="Watch",
                                         smc_gate_would_be_tier="Watch"))
        assert code in out["Decision_Reason_Codes"].split("|")
        assert f"DE_GATE=APPLIED:{reason}" in out["Decision_Trail"]

    def test_de_gate_on_but_not_triggered_is_pass(self):
        out = build_decision_trail(**_kw(de_gate_on=True, de_gate_reason=""))
        assert "DE_GATE=PASS" in out["Decision_Trail"] and out["Decision_Reason_Codes"] == ""

    def test_smc_gate_disabled_but_would_downgrade_is_reported_not_applied(self):
        out = build_decision_trail(**_kw(pre_gate_tier="Elite", final_tier="Elite", base_rank=R["Elite"],
                                         smc_gate_enabled=False, smc_gate_would_be_tier="Skip"))
        assert out["SMC_Gate_Changed_Tier"] is False and out["SMC_Gate_Would_Be_Tier"] == "Skip"
        assert "SMC_GATE_DISABLED_WOULD_DOWNGRADE" in out["Decision_Reason_Codes"]
        assert "SMC_GATE=DISABLED_WOULD_DOWNGRADE:Skip" in out["Decision_Trail"]

    def test_smc_gate_disabled_and_harmless(self):
        out = build_decision_trail(**_kw(smc_gate_enabled=False))
        assert "SMC_GATE=DISABLED" in out["Decision_Trail"] and out["Decision_Reason_Codes"] == ""

    def test_promotion_applicable_but_not_promoted(self):
        out = build_decision_trail(**_kw(promo_applicable=True, promo_promoted=False))
        assert "PROMOTION=NOT_PROMOTED" in out["Decision_Trail"]
        assert out["Decision_Reason_Codes"] == "PROMO_NOT_PROMOTED"

    def test_promoted_without_uplift_is_not_a_failure(self):
        out = build_decision_trail(**_kw(promo_applicable=True, promo_promoted=True, promo_tier="Actionable",
                                         promo_rank=R["Actionable"]))
        assert "PROMOTION=PROMOTED_NO_UPLIFT" in out["Decision_Trail"]
        assert "PROMO_NOT_PROMOTED" not in out["Decision_Reason_Codes"]

    def test_tied_sources_are_all_named(self):
        out = build_decision_trail(**_kw(natural_class="EXECUTE", base_rank=R["Execute"], natural_rank=R["Execute"],
                                         pre_gate_tier="Execute", final_tier="Execute",
                                         smc_gate_would_be_tier="Execute"))
        assert out["Final_Tier_Source"] == "BASE+NATURAL"

    def test_zero_rank_has_no_source(self):
        out = build_decision_trail(**_kw(cv4_base_tier="Skip", base_rank=0, pre_gate_tier="Skip", final_tier="Skip",
                                         smc_gate_would_be_tier="Skip", natural_class=None))
        assert out["Final_Tier_Source"] == "NONE" and "NATURAL=NONE" in out["Decision_Trail"]

    def test_every_emitted_code_is_in_the_stable_set(self):
        rng = random.Random(3)
        tiers = list(R)
        for _ in range(3000):
            pre, fin, would = (rng.choice(tiers) for _ in range(3))
            out = build_decision_trail(**_kw(
                cv4_base_tier=rng.choice(tiers), de_gate_on=rng.random() < .5,
                de_gate_reason=rng.choice(["", "hard_stop", "lifecycle=AVOID", "lifecycle=EXTENDED"]),
                natural_class=rng.choice([None, "ELITE", "EXECUTE", "WATCH"]),
                base_rank=rng.randint(0, 5), natural_rank=rng.choice([0, 4, 5]), promo_rank=rng.randint(0, 5),
                promo_applicable=rng.random() < .6, promo_promoted=rng.random() < .5, promo_bypassed=rng.random() < .5,
                promo_tier=rng.choice(tiers), pre_gate_tier=pre, final_tier=fin,
                smc_gate_enabled=rng.random() < .5, smc_gate_would_be_tier=would,
                action_raw=rng.choice(["a", "b"]), action_final=rng.choice(["a", "b"])))
            codes = [c for c in out["Decision_Reason_Codes"].split("|") if c]
            assert set(codes) <= DECISION_REASON_CODES
            assert len(codes) == len(set(codes))

    def test_pure_and_deterministic(self):
        kw = _kw(promo_applicable=True)
        assert build_decision_trail(**kw) == build_decision_trail(**kw)


# ──────────────────────────────────────────────────────────────────
#  2. score_stock integration
# ──────────────────────────────────────────────────────────────────
LOOSE = {"v4_actionable_leadership_min": 35, "v4_actionable_conviction_min": 25,
         "v4_actionable_entry_quality_min": 25, "v4_actionable_composite_min": 30}
VARIANTS = {
    "default": {}, "smc_gate_off": {"smc_structural_gate_enabled": False},
    "de_gate_on": {"ENABLE_STRUCTURAL_GATE": True}, "conflict_caps": {"smc_conflict_caps_tier": True},
    "loose_floors": {**LOOSE, "v4_execute_leadership_min": 45, "v4_execute_conviction_min": 35,
                     "v4_execute_entry_quality_min": 35, "v4_execute_composite_min": 40,
                     "v4_elite_leadership_min": 55, "v4_elite_conviction_min": 45,
                     "v4_elite_entry_quality_min": 45, "v4_elite_composite_min": 50},
    "loose_promo": {**LOOSE, "promo_execute_score_min": 0, "promo_elite_score_min": 25, "promo_min_rr_elite": 0.0,
                    "promo_bypass_enabled": True, "promo_bypass_min_leadership": 0},
    "loose_gate_off": {**LOOSE, "smc_structural_gate_enabled": False},
}


@pytest.fixture(scope="module")
def rows():
    logging.disable(logging.CRITICAL)
    nifty = T._nifty()
    g = np.random.default_rng(7)
    stocks = [T._stock(nifty, g, lead=(k % 2 == 0)) for k in range(40)]
    out = {}
    for name, extra in VARIANTS.items():
        out[name] = []
        for k, df in enumerate(stocks):
            s = {"prescreen_diagnostic": True}
            s.update(extra)
            out[name].append(se.score_stock(df, nifty, settings=s, symbol=f"S{k}"))
    yield out
    logging.disable(logging.NOTSET)


NEW_KEYS = ["Decision_Trail", "Decision_Reason_Codes", "Tier_CV4_Raw", "Pre_Gate_Tier",
            "SMC_Gate_Enabled", "SMC_Gate_Changed_Tier", "SMC_Gate_Would_Be_Tier", "Final_Tier_Source"]


class TestScoreStockIntegration:
    def test_new_fields_on_every_row_and_no_assembly_errors(self, rows):
        for var, rs in rows.items():
            for r in rs:
                assert all(k in r for k in NEW_KEYS), var
                assert "_tier_assembly_error" not in r, r.get("_tier_assembly_error")
                assert set(c for c in r["Decision_Reason_Codes"].split("|") if c) <= DECISION_REASON_CODES

    def test_trail_is_consistent_with_the_tier_columns(self, rows):
        for var, rs in rows.items():
            for r in rs:
                rec, pre = r["Recommendation"], r["Pre_Gate_Tier"]
                assert R[pre] >= R[rec], "the SMC gate may only downgrade"
                assert r["SMC_Gate_Changed_Tier"] == (pre != rec)
                if not r["SMC_Gate_Enabled"]:
                    assert rec == pre and r["SMC_Gate_Changed_Tier"] is False
                assert f"COMPOSE={pre}@" in r["Decision_Trail"]
                if r["SMC_Gate_Changed_Tier"]:
                    assert "SMC_GATE_DOWNGRADED" in r["Decision_Reason_Codes"]
                if r["Promoted"]:
                    assert "PROMOTION" in r["Final_Tier_Source"]
                assert r["Tier_CV4_Raw"] in R

    def test_action_cap_code_matches_the_action_columns(self, rows):
        for rs in rows.values():
            for r in rs:
                capped = r["Action"] != r["_action_raw"]
                assert ("ACTION_CAPPED_BY_TIER" in r["Decision_Reason_Codes"]) == capped

    def test_gate_disabled_variants_report_the_would_be_tier(self, rows):
        for var in ("smc_gate_off", "loose_gate_off"):
            for r in rows[var]:
                assert r["SMC_Gate_Enabled"] is False

    @pytest.mark.skipif(not os.path.exists(BASELINE), reason="baseline fixture missing")
    def test_tier_and_action_behaviour_identical_to_pre_change(self, rows):
        base = json.load(open(BASELINE))
        for var, rs in rows.items():
            for want, got in zip(base["variants"][var], rs):
                for k in base["keys"]:
                    assert repr(got.get(k)) == want[k], f"{var}:{k} moved"


# ──────────────────────────────────────────────────────────────────
#  3. failures are visible
# ──────────────────────────────────────────────────────────────────
def test_tier_assembly_failure_is_recorded_not_silent(monkeypatch):
    logging.disable(logging.CRITICAL)
    try:
        nifty = T._nifty()
        df = T._stock(nifty, np.random.default_rng(7), lead=True)

        def boom(*a, **k):
            raise RuntimeError("synthetic gate failure")

        monkeypatch.setattr(se, "apply_smc_structural_gate", boom)
        r = se.score_stock(df, nifty, settings={"prescreen_diagnostic": True}, symbol="X")
        assert r, "score_stock must still return a row (behaviour unchanged)"
        assert r["Decision_Reason_Codes"] == "TIER_ASSEMBLY_ERROR"
        assert "synthetic gate failure" in r["_tier_assembly_error"]
    finally:
        logging.disable(logging.NOTSET)
