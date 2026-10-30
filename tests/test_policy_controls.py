"""Tests for scripts/policy_controls.py (control policies; oracle verifier, no LLM)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import policy_controls as PC  # noqa: E402
import run_ragtruth_comparison as R  # noqa: E402
from src.claims.models import ClaimType  # noqa: E402
from src.controller.models import ControllerAction  # noqa: E402
from src.controller.policy import ControllerPolicyConfig  # noqa: E402
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter  # noqa: E402
from src.synthesis.grounding import support_for_claim  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "ragtruth_sample.json"
CFG = ControllerPolicyConfig(enable_repair=False)


def _prepared():
    ad = RAGTruthDatasetAdapter()
    ad.load_from_fixture(FIXTURE)
    return R.prepare(ad, ad.responses)[0]


def _run(prep, policy, budget=50):
    trace, _ = R.run_controller(prep, policy, budget, R.OracleAssessor(prep["gold"]), R._CostedMockRepairer(), "t")
    return trace


def _verify_order(prep, policy, budget=50):
    return [s.claim_id for s in _run(prep, policy, budget).steps if s.action == ControllerAction.VERIFY]


class TestBlindAndFit(unittest.TestCase):
    def test_blind_sets_constant_cts_without_touching_the_original(self):
        prepared = _prepared()
        before = [[c.triage_score for c in p["claims"]] for p in prepared]
        b = PC.blind(prepared)
        self.assertTrue(all(c.triage_score == 0.5 for p in b for c in p["claims"]))
        self.assertEqual(before, [[c.triage_score for c in p["claims"]] for p in prepared])
        self.assertEqual([p["gold"] for p in b], [p["gold"] for p in prepared])

    def test_logistic_fit_recovers_signs_and_improves_over_constant(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4000, 2))
        logit = 0.2 + 1.5 * x[:, 0] - 1.0 * x[:, 1]
        y = (rng.random(4000) < 1 / (1 + np.exp(-logit))).astype(float)
        w = PC.fit_logistic(np.column_stack([np.ones(4000), x]), y)
        self.assertGreater(w[1], 1.0)
        self.assertLess(w[2], -0.6)
        self.assertAlmostEqual(w[0], 0.2, delta=0.15)

    def test_fit_feature_ranker_shapes_and_determinism(self):
        prepared = _prepared()
        m1 = PC.fit_feature_ranker(prepared, ["cts", "type"])
        m2 = PC.fit_feature_ranker(prepared, ["cts", "type"])
        self.assertEqual(m1, m2)
        self.assertEqual(len(m1["weights"]), 3)  # bias + 2 features
        self.assertGreater(m1["n_train_claims"], 0)


class TestOrderingPolicies(unittest.TestCase):
    def test_feature_ranker_verifies_in_descending_score_order(self):
        prep = max(_prepared(), key=lambda p: len(p["claims"]))
        model = {"features": ["cts"], "weights": [0.0, 1.0]}  # score = CTS
        order = _verify_order(prep, PC.FeatureRanker(model, "R", CFG))
        cts = {c.claim_id: c.triage_score for c in prep["claims"]}
        self.assertEqual([cts[c] for c in order], sorted((cts[c] for c in order), reverse=True))
        neg = {"features": ["cts"], "weights": [0.0, -1.0]}
        order2 = _verify_order(prep, PC.FeatureRanker(neg, "R2", CFG))
        self.assertEqual([cts[c] for c in order2], sorted(cts[c] for c in order2))

    def test_lexical_support_verifies_least_supported_claim_first(self):
        prep = max(_prepared(), key=lambda p: len(p["claims"]))
        order = _verify_order(prep, PC.LexicalSupportPolicy(CFG))
        store = {c.claim_id: prep["evidence"][c.claim_id] for c in prep["claims"]}
        sup = {c.claim_id: support_for_claim(c.normalized_text, store[c.claim_id]) for c in prep["claims"]}
        score = {k: (1 - v["overlap"]) + (0.5 if v["missing_facts"] else 0.0) for k, v in sup.items()}
        self.assertEqual([score[c] for c in order], sorted((score[c] for c in order), reverse=True))

    def test_ties_fall_back_to_arrival_order(self):
        prep = max(_prepared(), key=lambda p: len(p["claims"]))
        const = PC.ScoredOrderPolicy(lambda c, i, n, s: 1.0, "const", CFG)
        self.assertEqual(_verify_order(prep, const), [c.claim_id for c in prep["claims"]])


class TestPositionMatchedRandom(unittest.TestCase):
    def test_profile_shifts_selection_toward_late_claims_and_is_reproducible(self):
        prep = max(_prepared(), key=lambda p: len(p["claims"]))
        first_pick = lambda profile, seed: _verify_order(prep, PC.PositionMatchedRandom(profile, seed, CFG), budget=1)[0]
        ids = [c.claim_id for c in prep["claims"]]
        late = [ids.index(first_pick([0, 0, 0, 0, 1], s)) for s in range(60)]
        early = [ids.index(first_pick([1, 0, 0, 0, 0], s)) for s in range(60)]
        self.assertGreater(np.mean(late), np.mean(early))
        self.assertEqual(first_pick([0.2, 0.2, 0.2, 0.2, 0.2], 5), first_pick([0.2, 0.2, 0.2, 0.2, 0.2], 5))

    def test_estimate_profile_from_rows(self):
        rows = [{"response_id": "r", "label": "SUPPORTED" if i < 2 else None} for i in range(5)]  # first two of five verified
        prof = PC.estimate_position_profile(rows, K=5)
        self.assertEqual(prof, [1.0, 1.0, 0.0, 0.0, 0.0])


class TestTypeThompson(unittest.TestCase):
    def _fb(self, policy, claim, label):
        from types import SimpleNamespace
        policy.record_feedback(claim=claim, action=ControllerAction.VERIFY, assessment=SimpleNamespace(label=label), action_failed=False)

    def test_learns_only_when_not_frozen(self):
        from src.evidence.models import AssessmentLabel
        claim = _prepared()[0]["claims"][0]
        ts = PC.TypeThompson(seed=0)
        a0, b0 = ts.posterior[claim.claim_type.value]
        self._fb(ts, claim, AssessmentLabel.CONTRADICTED)
        self.assertEqual(ts.posterior[claim.claim_type.value], [a0 + 1, b0])
        self._fb(ts, claim, AssessmentLabel.SUPPORTED)
        self.assertEqual(ts.posterior[claim.claim_type.value], [a0 + 1, b0 + 1])
        frozen = ts.frozen_copy(seed=1)
        before = dict(frozen.posterior)
        self._fb(frozen, claim, AssessmentLabel.CONTRADICTED)
        self.assertEqual(frozen.posterior, before)

    def test_prefers_the_type_with_the_higher_posterior(self):
        prep = max(_prepared(), key=lambda p: len({c.claim_type for c in p["claims"]}))
        types = sorted({c.claim_type for c in prep["claims"]}, key=lambda t: t.value)
        if len(types) < 2:
            self.skipTest("fixture has no response with two claim types")
        good, bad = types[0], types[1]
        post = {t.value: [1.0, 1.0] for t in ClaimType}
        post[good.value], post[bad.value] = [200.0, 1.0], [1.0, 200.0]
        picks = [next(c for c in prep["claims"] if c.claim_id == _verify_order(prep, PC.TypeThompson(post, seed=s, frozen=True, config=CFG), budget=1)[0]).claim_type
                 for s in range(30)]
        self.assertGreater(sum(p == good for p in picks), 25)


class TestAllControlsRespectBudgetAndNeverRepairWhenDisabled(unittest.TestCase):
    def test_budget_and_no_repair(self):
        model = {"features": ["cts", "type"], "weights": [0.0, 1.0, 1.0]}
        makers = [lambda: PC.FeatureRanker(model, "FR", CFG), lambda: PC.LexicalSupportPolicy(CFG),
                  lambda: PC.PositionMatchedRandom([0.2] * 5, 0, CFG), lambda: PC.TypeThompson(seed=0, frozen=True, config=CFG)]
        for prep in _prepared():
            for mk in makers:
                trace = _run(prep, mk(), budget=2)
                self.assertLessEqual(trace.total_cost.get("model_calls", 0), 2)
                self.assertFalse([s for s in trace.steps if s.action == ControllerAction.REPAIR])


class TestRegistryHook(unittest.TestCase):
    def test_registered_names_resolve_and_unknown_names_fall_back_to_the_benchmark_factory(self):
        import run_ragtruth_policy_controls as C
        from src.controller.policy import StaticAllocationPolicy
        C.REGISTRY["__test_lexical__"] = lambda seed, idx, er: PC.LexicalSupportPolicy(ControllerPolicyConfig(enable_repair=er))
        try:
            pol = R.make_policy("__test_lexical__", 0, 0, {}, False)
            self.assertEqual(pol.name, "LexicalSupport")
            self.assertFalse(pol.config.enable_repair)
            self.assertIsInstance(R.make_policy("StaticFIFO", 0, 0, {}, False), StaticAllocationPolicy)
            with self.assertRaises(ValueError):
                R.make_policy("definitely-not-a-policy", 0, 0, {}, False)
        finally:
            C.REGISTRY.pop("__test_lexical__", None)


if __name__ == "__main__":
    unittest.main()
