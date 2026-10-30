"""Tests for scripts/run_ragtruth_linucb_ablation.py (oracle control, no LLM)."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_ragtruth_comparison as R  # noqa: E402
import run_ragtruth_linucb_ablation as A  # noqa: E402
import src.controller.mab_policy as MP  # noqa: E402
from src.controller.mab_policy import ContextualMABPolicy, MABPolicyConfig  # noqa: E402
from src.controller.models import ControllerAction  # noqa: E402
from src.controller.policy import ControllerPolicyConfig  # noqa: E402
from src.controller.baselines import RandomPolicy  # noqa: E402
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "ragtruth_sample.json"


def _prepared():
    ad = RAGTruthDatasetAdapter()
    ad.load_from_fixture(FIXTURE)
    return R.prepare(ad, ad.responses)[0]


def _verify_order(prep, policy, budget=50):
    trace, _ = R.run_controller(prep, policy, budget, R.OracleAssessor(prep["gold"]), R._CostedMockRepairer(), "t")
    return [s.claim_id for s in trace.steps if s.action == ControllerAction.VERIFY]


class TestFeatureMask(unittest.TestCase):
    def test_mask_zeroes_only_requested_features_and_restores_the_function(self):
        original = MP.extract_context_features
        prep = _prepared()[0]
        claim = prep["claims"][0]
        claim.triage_score = 0.8

        from src.controller.budget import BudgetTracker
        from src.controller.models import PerClaimBudget, VerificationBudget
        bt = BudgetTracker(VerificationBudget(max_steps=5), PerClaimBudget())
        plain = MP.extract_context_features(claim, ControllerAction.VERIFY, {claim.claim_id: []}, {}, bt)
        with A.feature_mask([1, 2]):
            masked = MP.extract_context_features(claim, ControllerAction.VERIFY, {claim.claim_id: []}, {}, bt)
        self.assertEqual(masked[1], 0.0)
        self.assertEqual(masked[2], 0.0)
        self.assertEqual([m for i, m in enumerate(masked) if i not in (1, 2)], [p for i, p in enumerate(plain) if i not in (1, 2)])
        self.assertGreater(plain[1], 0.0)
        self.assertIs(MP.extract_context_features, original)  # restored

    def test_function_is_restored_even_if_the_block_raises(self):
        original = MP.extract_context_features
        with self.assertRaises(RuntimeError):
            with A.feature_mask([1]):
                raise RuntimeError("boom")
        self.assertIs(MP.extract_context_features, original)

    def test_without_cts_and_type_linucb_falls_back_to_cts_order_via_its_tiebreak(self):
        """Documented limitation: masking removes CTS from the learned score only; the tie-break (higher CTS first) remains."""
        for prep in _prepared():
            lin = ContextualMABPolicy(MABPolicyConfig(alpha=0.5, frozen=True, enable_repair=False))
            with A.feature_mask([1, 2]):
                order = _verify_order(prep, lin)
            ref = _verify_order(prep, R.make_policy("StaticCTSRanked", 0, 0, {}, enable_repair=False))
            self.assertEqual(order, ref)


class TestReferencePolicies(unittest.TestCase):
    def test_reverse_fifo_verifies_last_claim_first_and_low_cts_first_ascends(self):
        prep = max(_prepared(), key=lambda p: len(p["claims"]))
        n = len(prep["claims"])
        self.assertGreaterEqual(n, 2)
        rev = _verify_order(prep, A.OrderedPolicy("ReverseFIFO", ControllerPolicyConfig(enable_repair=False)))
        ids = [c.claim_id for c in prep["claims"]]
        self.assertEqual(rev, ids[::-1])
        low = _verify_order(prep, A.OrderedPolicy("LowCTSFirst", ControllerPolicyConfig(enable_repair=False)))
        cts = {c.claim_id: c.triage_score for c in prep["claims"]}
        self.assertEqual([cts[c] for c in low], sorted(cts[c] for c in low))

    def test_reference_policies_respect_budget_and_never_repair(self):
        for prep in _prepared():
            for name in ("ReverseFIFO", "LowCTSFirst"):
                trace, _ = R.run_controller(prep, A.OrderedPolicy(name, ControllerPolicyConfig(enable_repair=False)), 2,
                                            R.OracleAssessor(prep["gold"]), R._CostedMockRepairer(), "t")
                self.assertLessEqual(trace.total_cost.get("model_calls", 0), 2)
                self.assertFalse([s for s in trace.steps if s.action == ControllerAction.REPAIR])

    def test_run_setting_resolves_the_extra_reference_names(self):
        prepared = _prepared()
        res, rows = R.run_setting("ReverseFIFO", 3, 0, prepared, {}, None, R._CostedMockRepairer(), None,
                                  assessor_factory=lambda p: R.OracleAssessor(p["gold"]), enable_repair=False, n_boot=0)
        self.assertEqual(res["policy"], "ReverseFIFO")
        self.assertNotIn("REPAIR", res["action_counts"])


class TestAblationScriptSmoke(unittest.TestCase):
    def test_runs_on_fixture_with_untrained_policies_and_reports_variants_and_references(self):
        with tempfile.TemporaryDirectory() as d:
            out = A.main(["--fixture", str(FIXTURE), "--seeds", "0,1", "--budgets", "2,50", "--train-responses", "0",
                          "--n-boot", "50", "--out-dir", d, "--tag", "t"])
            b = out["budgets"]["50"]["policies"]
            self.assertEqual(set(b), set(A.VARIANTS) | set(A.REFERENCES))
            self.assertEqual(b["StaticFIFO"]["selection_recall"], 1.0)  # unlimited budget selects every positive
            self.assertEqual(b["full"]["n_runs"], 2)
            self.assertIn("full - no_CTS", out["budgets"]["2"]["paired"])
            self.assertIn("full - ReverseFIFO", out["budgets"]["2"]["paired"])
            self.assertTrue(any(Path(d).glob("*.md")))


if __name__ == "__main__":
    unittest.main()
