"""Tests for the Random and VerifyAll baselines (ordering, budget, seed reproducibility)."""

import unittest

from src.claims.models import ClaimRecord, ClaimType, DecompositionStatus
from src.controller.baselines import RandomPolicy, VerifyAllPolicy
from src.controller.controller import VerificationController
from src.controller.interfaces import MockRepairer, MockRetriever
from src.controller.models import ControllerAction, VerificationBudget
from src.evidence.assessor import MockDeterministicAssessor
from src.evidence.models import EvidencePassage

_PASSAGE = EvidencePassage(
    evidence_id="ev", doc_title="Doc",
    source_text="Arthur's Magazine was not published in Philadelphia. It was based in Boston.",
    retrieval_score=0.9,
)


def _claims(n=4):
    t = "Arthur's Magazine was published in Philadelphia."
    return [
        ClaimRecord(claim_id=f"c{i}", source_sentence_id=f"s{i}", original_text=t, normalized_text=t,
                    claim_type=ClaimType.FACTUAL_RELATION, decomposition_status=DecompositionStatus.ATOMIC,
                    triage_score=0.9 - 0.1 * i)
        for i in range(n)
    ]


def _run(policy, calls=10, n=4):
    ctl = VerificationController(
        retriever=MockRetriever(canned_evidence={f"c{i}": [_PASSAGE] for i in range(n)}),
        assessor=MockDeterministicAssessor(), repairer=MockRepairer(), policy=policy,
        global_budget=VerificationBudget(max_steps=40, max_model_calls=calls),
    )
    return ctl.run(_claims(n), initial_evidence={f"c{i}": [_PASSAGE] for i in range(n)})


def _actions(trace):
    return [(s.action.value, s.claim_id) for s in trace.steps if s.action != ControllerAction.STOP]


class TestVerifyAll(unittest.TestCase):
    def test_all_verifies_precede_any_repair(self):
        acts = _actions(_run(VerifyAllPolicy()))
        first_repair = next(i for i, (a, _) in enumerate(acts) if a == "REPAIR")
        self.assertEqual([a for a, _ in acts[:first_repair]], ["VERIFY"] * 4)
        self.assertEqual([c for _, c in acts[:4]], ["c0", "c1", "c2", "c3"])


class TestRandom(unittest.TestCase):
    def test_same_seed_same_trace_different_seed_can_differ(self):
        a = _actions(_run(RandomPolicy(seed=1)))
        b = _actions(_run(RandomPolicy(seed=1)))
        self.assertEqual(a, b)
        others = {tuple(_actions(_run(RandomPolicy(seed=s)))) for s in range(2, 8)}
        self.assertGreater(len(others | {tuple(a)}), 1)

    def test_only_verifies_and_repairs_never_repairs_unverified(self):
        acts = _actions(_run(RandomPolicy(seed=3)))
        verified = set()
        for a, cid in acts:
            if a == "REPAIR":
                self.assertIn(cid, verified)
            if a == "VERIFY":
                verified.add(cid)


class TestCTSPoliciesDivergeOnTies(unittest.TestCase):
    """DeterministicCTS and StaticCTSRanked share their CTS ordering, so they only differ on exact
    CTS ties where the tied claims' eligible actions have different priorities. This pins down both
    facts: identical on distinct scores, divergent on the constructed tie."""

    def _run(self, policy, triage):
        from src.evidence.models import AssessmentLabel, EvidenceAssessment, VerificationCost

        class Scripted(MockDeterministicAssessor):
            def assess(self, claim, passages):  # c0 -> INSUFFICIENT (needs RETRIEVE next), others SUPPORTED
                lab = AssessmentLabel.INSUFFICIENT if claim.claim_id == "c0" else AssessmentLabel.SUPPORTED
                return EvidenceAssessment(claim_id=claim.claim_id, evidence_ids=["ev"], supporting_spans={},
                                          label=lab, confidence=0.9, rationale="scripted",
                                          cost=VerificationCost(model_calls=1, latency_seconds=0.0))

        claims = _claims(2)
        for c, t in zip(claims, triage):
            c.triage_score = t
        ev = {f"c{i}": [_PASSAGE] for i in range(2)}
        ctl = VerificationController(
            retriever=MockRetriever(canned_evidence=ev), assessor=Scripted(), repairer=MockRepairer(), policy=policy,
            global_budget=VerificationBudget(max_steps=10, max_model_calls=10),
        )
        return _actions(ctl.run(claims, initial_evidence=ev))

    def test_identical_when_scores_are_distinct(self):
        from src.controller.policy import SequentialDecisionPolicy, StaticCTSRankedPolicy
        a = self._run(SequentialDecisionPolicy(), [0.9, 0.5])
        b = self._run(StaticCTSRankedPolicy(), [0.9, 0.5])
        self.assertEqual(a, b)

    def test_diverge_on_exact_tie_with_different_action_priorities(self):
        from src.controller.policy import SequentialDecisionPolicy, StaticCTSRankedPolicy
        a = self._run(SequentialDecisionPolicy(), [0.7, 0.7])
        b = self._run(StaticCTSRankedPolicy(), [0.7, 0.7])
        self.assertNotEqual(a, b, (a, b))


if __name__ == "__main__":
    unittest.main()
