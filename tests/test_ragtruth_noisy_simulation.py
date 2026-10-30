"""Tests for the noisy-verifier SIMULATION (scripts/run_ragtruth_noisy_simulation.py). No LLM, no real verifier."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_ragtruth_comparison as R  # noqa: E402
import run_ragtruth_noisy_simulation as N  # noqa: E402
from src.evaluation import ragtruth_metrics as M  # noqa: E402
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "ragtruth_sample.json"
CKPT = ROOT / "experiments" / "runs" / "linucb_warmstart_train100_300.json"


def _claim(cid):
    return SimpleNamespace(claim_id=cid)


def _flagged(assessor, cid):
    return assessor.assess(_claim(cid), []).label.value == "CONTRADICTED"


class TestSimulatedAssessor(unittest.TestCase):
    def setUp(self):
        self.gold = {f"c{i}": ("HALLUCINATED" if i % 2 == 0 else "UNFLAGGED") for i in range(20000)}
        self.gold.update({"amb": "UNCERTAIN"})

    def test_empirical_rates_match_targets(self):
        a = N.SimulatedAssessor(self.gold, "r1", tpr=0.30, fpr=0.10, noise_seed=0)
        pos = [_flagged(a, c) for c, g in self.gold.items() if g == "HALLUCINATED"]
        neg = [_flagged(a, c) for c, g in self.gold.items() if g == "UNFLAGGED"]
        self.assertAlmostEqual(sum(pos) / len(pos), 0.30, delta=0.015)
        self.assertAlmostEqual(sum(neg) / len(neg), 0.10, delta=0.015)

    def test_deterministic_and_independent_of_call_order_and_instance(self):
        a1 = N.SimulatedAssessor(self.gold, "r1", 0.3, 0.1, 5)
        a2 = N.SimulatedAssessor(self.gold, "r1", 0.3, 0.1, 5)
        keys = list(self.gold)[:500]
        fwd = [_flagged(a1, c) for c in keys]
        rev = [_flagged(a2, c) for c in reversed(keys)][::-1]
        self.assertEqual(fwd, rev)
        self.assertEqual(fwd, [_flagged(a1, c) for c in keys])  # repeat call -> same verdict

    def test_noise_seed_and_response_id_change_the_draws(self):
        keys = [c for c, g in self.gold.items() if g == "UNFLAGGED"][:2000]
        base = [_flagged(N.SimulatedAssessor(self.gold, "r1", 0.3, 0.3, 0), c) for c in keys]
        self.assertNotEqual(base, [_flagged(N.SimulatedAssessor(self.gold, "r1", 0.3, 0.3, 1), c) for c in keys])
        self.assertNotEqual(base, [_flagged(N.SimulatedAssessor(self.gold, "r2", 0.3, 0.3, 0), c) for c in keys])

    def test_boundary_rates_and_ambiguous_gold_treated_as_negative(self):
        never = N.SimulatedAssessor(self.gold, "r", 0.0, 0.0, 0)
        always = N.SimulatedAssessor(self.gold, "r", 1.0, 1.0, 0)
        self.assertFalse(any(_flagged(never, c) for c in self.gold))
        self.assertTrue(all(_flagged(always, c) for c in self.gold))
        only_pos = N.SimulatedAssessor(self.gold, "r", 1.0, 0.0, 0)
        self.assertFalse(_flagged(only_pos, "amb"))  # ambiguous gold gets the FPR (here 0), not the TPR
        self.assertTrue(_flagged(only_pos, "c0"))

    def test_costs_one_model_call_like_a_real_verification(self):
        r = N.SimulatedAssessor(self.gold, "r", 0.3, 0.1, 0).assess(_claim("c0"), [])
        self.assertEqual(r.cost.model_calls, 1)


class TestSimulationThroughHarness(unittest.TestCase):
    def setUp(self):
        ad = RAGTruthDatasetAdapter()
        ad.load_from_fixture(FIXTURE)
        self.prepared, _ = R.prepare(ad, ad.responses)

    def _run(self, policy, tpr, fpr, ns=0, budget=50):
        res, rows = R.run_setting(policy, budget, 0, self.prepared, {}, None, R._CostedMockRepairer(), None,
                                  assessor_factory=N.sim_factory(tpr, fpr, ns), enable_repair=False, n_boot=0)
        return res, rows

    def test_same_claim_gets_the_same_simulated_verdict_under_every_policy(self):
        labels = {}
        for pol in ("StaticFIFO", "StaticCTSRanked", "VerifyAll"):
            _, rows = self._run(pol, 0.5, 0.5)
            labels[pol] = {(r["response_id"], r["claim_id"]): r["label"] for r in rows}
        for k in labels["StaticFIFO"]:  # unlimited budget: all claims verified by all policies
            self.assertEqual(len({labels[p][k] for p in labels}), 1, k)

    def test_zero_tpr_gives_zero_detection_and_full_tpr_perfect_recall(self):
        res0, _ = self._run("VerifyAll", 0.0, 0.0)
        self.assertEqual(res0["claim_broad"]["tp"], 0)
        res1, _ = self._run("VerifyAll", 1.0, 0.0)
        self.assertEqual(res1["claim_broad"]["recall"], 1.0)
        self.assertEqual(res1["claim_broad"]["fp"], 0)
        self.assertEqual(res1["selection"]["selection_recall"], 1.0)

    def test_detection_equals_flagged_share_of_verified_positives(self):
        res, rows = self._run("VerifyAll", 1.0, 1.0)
        self.assertEqual(res["claim_broad"]["tn"], 0)  # everything flagged -> no true negatives
        self.assertEqual(res["claim_broad"]["tp"], res["selection"]["verified_positives"])


class TestVectorisedBootstrapHelpers(unittest.TestCase):
    def test_metric_values_match_scalar_definitions(self):
        A = {k: np.array([3.0, 1.0]) for k in N.ARR_KEYS}
        m = N.metric_values(A)
        self.assertAlmostEqual(m["selection_recall"], 4 / 8)
        self.assertAlmostEqual(m["detection_recall"], 4 / 8)
        self.assertAlmostEqual(m["detection_precision"], 4 / 8)
        self.assertAlmostEqual(m["detection_f1"], 0.5)
        boots = N.metric_values(A, np.zeros((5, 2), dtype=int))  # resample response 0 only
        self.assertTrue(np.allclose(boots["selection_recall"], 0.5))

    def test_zero_detection_gives_zero_f1_not_nan_and_empty_ci_none(self):
        A = {k: np.zeros(3) for k in N.ARR_KEYS}
        A["bfp"] = np.array([1.0, 1.0, 1.0]); A["bfn"] = np.array([1.0, 0.0, 1.0])
        self.assertEqual(N.metric_values(A)["detection_f1"], 0.0)
        self.assertIsNone(N.ci(np.array([np.nan] * 30)))


@unittest.skipUnless(CKPT.exists(), "checkpoint not present")
class TestScriptSmoke(unittest.TestCase):
    def test_runs_end_to_end_on_fixture_and_labels_itself_a_simulation(self):
        with tempfile.TemporaryDirectory() as d:
            out = N.main(["--fixture", str(FIXTURE), "--scenarios", "S1_measured_readable_evidence,S3_hypothetical_working_verifier",
                          "--noise-seeds", "0,1", "--budgets", "2,50", "--random-seeds", "0,1", "--train-responses", "0",
                          "--linucb-oracle", str(CKPT), "--n-boot", "50", "--out-dir", d, "--tag", "t"])
            self.assertIn("SIMULATION", out["manifest"])
            self.assertFalse(out["manifest"]["repair_enabled"])
            self.assertTrue((Path(d) / (out["manifest"]["run_id"] + ".md")).read_text(encoding="utf-8").startswith("# "))
            s1 = out["scenarios"]["S1_measured_readable_evidence"]["budgets"]["50"]["policies"]
            self.assertEqual(s1["VerifyAll"]["selection_recall"], 1.0)  # unlimited budget verifies every positive
            self.assertIn("Random(mean of seeds)", s1)
            self.assertIn("LinUCB_oracle_trained - VerifyAll [selection_recall]",
                          out["scenarios"]["S1_measured_readable_evidence"]["budgets"]["50"]["paired"])


if __name__ == "__main__":
    unittest.main()
