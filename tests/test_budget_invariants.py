"""Budget boundary invariants across all policies, with costed (1 model call) assessor/repairer.

The mock assessor/repairer report 0 model calls, so budget overshoot is invisible with them.
These fixtures charge 1 call each, as the real LLM components do.
"""

import itertools
import unittest

from src.controller.controller import VerificationController
from src.controller.interfaces import MockRepairer, MockRetriever
from src.controller.mab_policy import ContextualMABPolicy, MABPolicyConfig
from src.controller.models import VerificationBudget
from src.controller.policy import SequentialDecisionPolicy, StaticAllocationPolicy, StaticCTSRankedPolicy
from src.evidence.assessor import MockDeterministicAssessor
from src.evidence.models import EvidencePassage, VerificationCost
from src.claims.models import ClaimRecord, ClaimType, DecompositionStatus


class _CostedAssessor(MockDeterministicAssessor):
    def assess(self, claim, passages):
        a = super().assess(claim, passages)
        a.cost = VerificationCost(model_calls=1, latency_seconds=0.01)
        return a


class _CostedRepairer(MockRepairer):
    def repair(self, claim, assessment, passages):
        r, _ = super().repair(claim, assessment, passages)
        return r, VerificationCost(model_calls=1, latency_seconds=0.01)


def _claims(n=3):
    text = "Arthur's Magazine was published in Philadelphia."
    return [
        ClaimRecord(
            claim_id=f"c{i}", source_sentence_id=f"s{i}", original_text=text, normalized_text=text,
            claim_type=ClaimType.FACTUAL_RELATION, decomposition_status=DecompositionStatus.ATOMIC,
            triage_score=0.9 - 0.1 * i,
        )
        for i in range(n)
    ]


_PASSAGE = EvidencePassage(
    evidence_id="ev", doc_title="Arthur's Magazine",
    source_text="Arthur's Magazine was not published in Philadelphia. It was based in Boston.",
    retrieval_score=0.9,
)

_POLICIES = {
    "deterministic_cts": SequentialDecisionPolicy,
    "fifo": StaticAllocationPolicy,
    "static_cts": StaticCTSRankedPolicy,
    "linucb": lambda: ContextualMABPolicy(MABPolicyConfig(frozen=True)),
}


class TestBudgetInvariants(unittest.TestCase):
    def test_no_policy_exceeds_step_or_model_call_budget(self):
        for (name, mk), max_calls, max_steps in itertools.product(_POLICIES.items(), range(0, 8), range(1, 8)):
            with self.subTest(policy=name, max_calls=max_calls, max_steps=max_steps):
                ctl = VerificationController(
                    retriever=MockRetriever(canned_evidence={f"c{i}": [_PASSAGE] for i in range(3)}),
                    assessor=_CostedAssessor(), repairer=_CostedRepairer(), policy=mk(),
                    global_budget=VerificationBudget(max_steps=max_steps, max_model_calls=max_calls),
                )
                trace = ctl.run(_claims())
                self.assertLessEqual(trace.total_steps, max_steps)
                self.assertLessEqual(trace.total_cost.get("model_calls", 0), max_calls)

    def test_zero_model_call_budget_makes_no_llm_calls(self):
        ctl = VerificationController(
            retriever=MockRetriever(canned_evidence={f"c{i}": [_PASSAGE] for i in range(3)}),
            assessor=_CostedAssessor(), repairer=_CostedRepairer(),
            global_budget=VerificationBudget(max_steps=10, max_model_calls=0),
        )
        trace = ctl.run(_claims())
        self.assertEqual(trace.total_cost.get("model_calls", 0), 0)


if __name__ == "__main__":
    unittest.main()
