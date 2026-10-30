"""Comprehensive unit tests for Stage B Sequential Verification Controller.

All tests are deterministic, offline, and require no live LLM, Qdrant, or network access.

Coverage includes:
1. Single claim selected and verified within budget.
2. Multiple claims competing for a limited budget.
3. CTS influencing priority without acting as a falsity probability.
4. New evidence dynamically changing subsequent decisions.
5. Supported claims not being repeatedly verified.
6. Insufficient evidence triggering additional retrieval.
7. Contradictory evidence triggering repair/qualification.
8. Conflicted evidence distinguished from definitive contradiction.
9. Repair recording original claim, proposed revision, and evidence provenance.
10. Deferred claims remaining strictly unverified.
11. Reconsideration of deferred claims under remaining budget.
12. Global budget exhaustion across distinct units (steps, model calls, tokens).
13. Per-claim budget enforcement (max retrievals, verifications, repairs).
14. Accurate cost aggregation across actions.
15. Unknown/None cost handling.
16. Failed actions and exception handling without unhandled crash.
17. Maximum-step stopping condition.
18. Deterministic tie-breaking across runs.
19. Empty claims and zero budget edge cases.
20. Duplicate claim ID detection.
21. Preservation of valid ClaimState lifecycle transitions.
22. SUPPORTED label alone not causing VERIFIED transition without threshold.
23. Complete, valid JSON serialization of controller traces.
"""

import json
import unittest
from typing import List, Optional

from src.claims.models import (
    ClaimRecord,
    ClaimState,
    ClaimType,
    DecompositionStatus,
    InvalidStateTransitionError,
)
from src.controller.budget import BudgetTracker
from src.controller.controller import VerificationController
from src.controller.interfaces import BaseRetriever, MockRepairer, MockRetriever
from src.controller.models import (
    ControllerAction,
    ControllerTrace,
    PerClaimBudget,
    RepairResult,
    StoppingReason,
    VerificationBudget,
)
from src.controller.policy import ControllerPolicyConfig, SequentialDecisionPolicy
from src.evidence.assessor import BaseAssessor, MockDeterministicAssessor
from src.evidence.models import (
    AssessmentLabel,
    EvidenceAssessment,
    EvidencePassage,
    VerificationCost,
)


def _make_claim(
    claim_id: str,
    text: str,
    claim_type: ClaimType = ClaimType.FACTUAL_RELATION,
    triage_score: Optional[float] = 0.5,
    state: ClaimState = ClaimState.PENDING,
) -> ClaimRecord:
    """Create a ClaimRecord fixture for controller tests."""
    c = ClaimRecord(
        claim_id=claim_id,
        source_sentence_id="s_test",
        original_text=text,
        normalized_text=text,
        claim_type=claim_type,
        decomposition_status=DecompositionStatus.ATOMIC,
        state=state,
        triage_score=triage_score,
    )
    return c


def _make_passage(
    evidence_id: str,
    title: str,
    text: str,
    score: float = 0.90,
) -> EvidencePassage:
    """Create an EvidencePassage fixture."""
    return EvidencePassage(
        evidence_id=evidence_id,
        doc_title=title,
        source_text=text,
        retrieval_score=score,
    )


class TestSingleClaimVerification(unittest.TestCase):
    """Test standard happy-path verification of a single claim."""

    def test_single_claim_retrieved_and_verified(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.", triage_score=0.8)
        passage = _make_passage(
            "ev1", "Arthur's Magazine",
            "Arthur's Magazine was published in Philadelphia from 1844 to 1846."
        )

        retriever = MockRetriever(canned_evidence={"c1": [passage]})
        assessor = MockDeterministicAssessor()
        controller = VerificationController(
            retriever=retriever,
            assessor=assessor,
            global_budget=VerificationBudget(max_steps=10),
        )

        trace = controller.run([claim])

        self.assertEqual(claim.state, ClaimState.VERIFIED)
        self.assertIsNotNone(claim.evidence_provenance)
        self.assertEqual(trace.stopping_reason, StoppingReason.ALL_CLAIMS_RESOLVED)
        self.assertEqual(trace.claims_summary["verified"], 1)

        # Check trace actions sequence: RETRIEVE -> VERIFY -> STOP
        actions = [s.action for s in trace.steps]
        self.assertIn(ControllerAction.RETRIEVE.value, actions)
        self.assertIn(ControllerAction.VERIFY.value, actions)
        self.assertEqual(actions[-1], ControllerAction.STOP.value)


class TestMultipleClaimsCompetition(unittest.TestCase):
    """Test priority competition when budget is insufficient for all claims."""

    def test_high_priority_claim_verified_first_low_priority_deferred(self):
        c_high = _make_claim("c_high", "High priority claim text.", triage_score=0.95)
        c_low = _make_claim("c_low", "Low priority claim text.", triage_score=0.10)

        ev_high = _make_passage("ev_h", "Doc High", "High priority claim text verified here.")
        ev_low = _make_passage("ev_l", "Doc Low", "Low priority claim text verified here.")

        retriever = MockRetriever(canned_evidence={
            "c_high": [ev_high],
            "c_low": [ev_low],
        })
        assessor = MockDeterministicAssessor()

        # Strict budget: only 2 action steps (enough to RETRIEVE and VERIFY 1 claim)
        controller = VerificationController(
            retriever=retriever,
            assessor=assessor,
            global_budget=VerificationBudget(max_steps=2),
        )

        trace = controller.run([c_low, c_high])  # c_low passed first in list to test sorting

        # c_high must be processed before c_low despite order in list
        first_claim_id = trace.steps[0].claim_id
        self.assertEqual(first_claim_id, "c_high")
        self.assertEqual(c_high.state, ClaimState.VERIFIED)
        # c_low was not processed and remained PENDING or DEFERRED
        self.assertNotEqual(c_low.state, ClaimState.VERIFIED)
        self.assertEqual(trace.stopping_reason, StoppingReason.MAX_STEPS_REACHED)


class TestCTSPriorityBehavior(unittest.TestCase):
    """Verify that CTS dictates scheduling priority without prejudging truth value."""

    def test_cts_determines_selection_order(self):
        c1 = _make_claim("c1", "Claim one.", triage_score=0.30)
        c2 = _make_claim("c2", "Claim two.", triage_score=0.85)
        c3 = _make_claim("c3", "Claim three.", triage_score=0.60)

        policy = SequentialDecisionPolicy()
        budget = BudgetTracker(VerificationBudget(max_steps=10))

        action, selected_claim, _ = policy.select_next_action(
            claims=[c1, c2, c3],
            evidence_store={},
            assessments={},
            repairs={},
            budget_tracker=budget,
        )

        # c2 has highest CTS (0.85) and should be selected first
        self.assertEqual(selected_claim.claim_id, "c2")
        self.assertEqual(action, ControllerAction.RETRIEVE)


class TestDynamicEvidenceResponse(unittest.TestCase):
    """Test that new evidence dynamically updates the next chosen action."""

    def test_retrieval_triggers_verification_at_next_step(self):
        claim = _make_claim("c1", "Some claim text.", triage_score=0.7)
        policy = SequentialDecisionPolicy()
        budget = BudgetTracker(VerificationBudget(max_steps=10))

        # Initially no evidence -> Action must be RETRIEVE
        action1, _, _ = policy.select_next_action(
            [claim], evidence_store={"c1": []}, assessments={}, repairs={}, budget_tracker=budget
        )
        self.assertEqual(action1, ControllerAction.RETRIEVE)

        # After evidence is added -> Action must dynamically switch to VERIFY
        passage = _make_passage("ev1", "Doc", "Some claim text.")
        action2, _, _ = policy.select_next_action(
            [claim], evidence_store={"c1": [passage]}, assessments={}, repairs={}, budget_tracker=budget
        )
        self.assertEqual(action2, ControllerAction.VERIFY)


class TestSupportedClaimNotRepeatedlyVerified(unittest.TestCase):
    """A claim already verified/supported should not be re-verified endlessly."""

    def test_verified_claim_is_terminal(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.", state=ClaimState.VERIFIED)
        claim.evidence_provenance = {"verified": True}

        policy = SequentialDecisionPolicy()
        budget = BudgetTracker(VerificationBudget(max_steps=10))

        action, target, _ = policy.select_next_action(
            [claim], evidence_store={}, assessments={}, repairs={}, budget_tracker=budget
        )
        self.assertEqual(action, ControllerAction.STOP)
        self.assertIsNone(target)


class TestInsufficientEvidenceHandling(unittest.TestCase):
    """Test handling of inconclusive evidence and bounded retries."""

    def test_insufficient_triggers_retrieval_then_unresolved(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        # Evidence that does not substantiate the claim
        irrelevant_passage = _make_passage("ev_irr", "Doc", "Quantum physics is a branch of science.")

        retriever = MockRetriever(canned_evidence={"c1": [irrelevant_passage]})
        assessor = MockDeterministicAssessor()

        # Limit per-claim retrievals to 2
        per_claim = PerClaimBudget(max_retrievals=2)
        controller = VerificationController(
            retriever=retriever,
            assessor=assessor,
            global_budget=VerificationBudget(max_steps=10),
            per_claim_budget=per_claim,
        )

        trace = controller.run([claim])

        # Claim could not be substantiated within retrieval limits -> UNRESOLVED
        self.assertEqual(claim.state, ClaimState.UNRESOLVED)
        self.assertEqual(trace.claims_summary["unresolved"], 1)


class TestContradictionAndRepair(unittest.TestCase):
    """Test contradiction detection and structured repair execution."""

    def test_contradicted_claim_triggers_repair(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        # Evidence explicitly contradicting Philadelphia
        contra_passage = _make_passage(
            "ev_contra", "Arthur's Magazine",
            "Arthur's Magazine was not published in Philadelphia. It was based in Boston."
        )

        retriever = MockRetriever(canned_evidence={"c1": [contra_passage]})
        assessor = MockDeterministicAssessor()
        repairer = MockRepairer()

        controller = VerificationController(
            retriever=retriever,
            assessor=assessor,
            repairer=repairer,
            global_budget=VerificationBudget(max_steps=10),
        )

        trace = controller.run([claim])

        # A repair action should have been taken
        actions = [s.action for s in trace.steps]
        self.assertIn(ControllerAction.REPAIR.value, actions)
        # Repaired claim transitions to UNRESOLVED (not falsely verified)
        self.assertEqual(claim.state, ClaimState.UNRESOLVED)
        self.assertIn("Boston", contra_passage.source_text)

    def test_repair_is_reverified_and_charged_to_budget(self):
        """After REPAIR the revised text is re-assessed; the outcome is recorded and costed."""
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        passage = _make_passage(
            "ev_contra", "Arthur's Magazine",
            "Arthur's Magazine was not published in Philadelphia. It was based in Boston."
        )
        seen_texts = []

        class RecordingAssessor(MockDeterministicAssessor):
            def assess(self, claim, passages):
                seen_texts.append(claim.normalized_text)
                return super().assess(claim, passages)

        controller = VerificationController(
            retriever=MockRetriever(canned_evidence={"c1": [passage]}),
            assessor=RecordingAssessor(),
            repairer=MockRepairer(),
            global_budget=VerificationBudget(max_steps=10),
        )
        trace = controller.run([claim])

        repair_steps = [s for s in trace.steps if s.action == ControllerAction.REPAIR.value]
        self.assertEqual(len(repair_steps), 1)
        rv = repair_steps[0].repair_data["reverification"]
        self.assertIsNotNone(rv)
        self.assertIn("verified", rv)
        # The assessor was called on the revised text, not only the original claim
        revised = repair_steps[0].repair_data["proposed_revision"]
        self.assertIn(revised, seen_texts)
        # Original claim text is left untouched
        self.assertEqual(claim.normalized_text, "Arthur's Magazine was published in Philadelphia.")

    def test_reverification_skipped_when_budget_exhausted(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        passage = _make_passage(
            "ev_contra", "Arthur's Magazine",
            "Arthur's Magazine was not published in Philadelphia. It was based in Boston."
        )
        controller = VerificationController(
            retriever=MockRetriever(canned_evidence={"c1": [passage]}),
            assessor=MockDeterministicAssessor(),
            repairer=MockRepairer(),
            global_budget=VerificationBudget(max_steps=3),
        )
        trace = controller.run([claim])
        self.assertEqual(trace.total_steps, 3)  # budget respected; trailing STOP entry is not a budgeted step
        for s in trace.steps:
            if s.action == ControllerAction.REPAIR.value:
                # no step budget left for a re-check -> must be recorded as not attempted
                self.assertIsNone(s.repair_data["reverification"])


class TestConflictedEvidenceDistinction(unittest.TestCase):
    """Conflicted evidence from opposing sources must be distinguished from definitive contradiction."""

    def test_conflicted_evidence_handled_properly(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        p_support = _make_passage("ev_s", "Doc A", "Arthur's Magazine was published in Philadelphia.")
        p_contra = _make_passage("ev_c", "Doc B", "Arthur's Magazine was not published in Philadelphia.")

        retriever = MockRetriever(canned_evidence={"c1": [p_support, p_contra]})
        assessor = MockDeterministicAssessor()
        repairer = MockRepairer()

        controller = VerificationController(
            retriever=retriever,
            assessor=assessor,
            repairer=repairer,
            global_budget=VerificationBudget(max_steps=10),
        )

        trace = controller.run([claim])

        # Assessment should have been CONFLICTED
        verify_steps = [s for s in trace.steps if s.action == ControllerAction.VERIFY.value]
        self.assertTrue(any(s.assessment_label == AssessmentLabel.CONFLICTED.value for s in verify_steps))


class TestDeferredClaimsLifecycle(unittest.TestCase):
    """Test deferral when budget is constrained, and subsequent reconsideration."""

    def test_deferred_claim_remains_unverified(self):
        claim = _make_claim("c1", "Low priority claim text.", triage_score=0.1)

        # Budget allows 0 steps
        budget = VerificationBudget(max_steps=0)
        controller = VerificationController(global_budget=budget)

        trace = controller.run([claim])
        self.assertNotEqual(claim.state, ClaimState.VERIFIED)

    def test_reconsideration_of_deferred_claim(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.", state=ClaimState.DEFERRED)
        passage = _make_passage("ev1", "Doc", "Arthur's Magazine was published in Philadelphia.")

        retriever = MockRetriever(canned_evidence={"c1": [passage]})
        assessor = MockDeterministicAssessor()

        controller = VerificationController(
            retriever=retriever,
            assessor=assessor,
            global_budget=VerificationBudget(max_steps=10),
            policy=SequentialDecisionPolicy(ControllerPolicyConfig(allow_reconsideration=True)),
        )

        trace = controller.run([claim])
        self.assertEqual(claim.state, ClaimState.VERIFIED)


class TestBudgetEnforcement(unittest.TestCase):
    """Test global and per-claim budget enforcement across distinct dimensions."""

    def test_global_model_call_budget_exhaustion(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        passage = _make_passage("ev1", "Doc", "Arthur's Magazine was published in Philadelphia.")

        retriever = MockRetriever(canned_evidence={"c1": [passage]})
        assessor = MockDeterministicAssessor()

        # Custom cost reporting 1 model call per verification
        class CostAssessor(BaseAssessor):
            @property
            def assessor_id(self) -> str:
                return "cost_assessor"

            def assess(self, claim, evidence):
                res = assessor.assess(claim, evidence)
                res.cost = VerificationCost(model_calls=1, latency_seconds=0.01)
                return res

        # Global budget: exactly 1 model call allowed
        controller = VerificationController(
            retriever=retriever,
            assessor=CostAssessor(),
            global_budget=VerificationBudget(max_model_calls=1, max_steps=10),
        )

        c2 = _make_claim("c2", "Second claim text.", triage_score=0.4)
        trace = controller.run([claim, c2])

        self.assertEqual(trace.stopping_reason, StoppingReason.BUDGET_EXHAUSTED)
        self.assertLessEqual(trace.total_cost["model_calls"], 1)

    def test_per_claim_retrieval_budget_enforcement(self):
        tracker = BudgetTracker(
            global_budget=VerificationBudget(max_steps=20),
            per_claim_budget=PerClaimBudget(max_retrievals=1),
        )

        can_afford1, _ = tracker.can_afford(ControllerAction.RETRIEVE, "c1")
        self.assertTrue(can_afford1)
        tracker.record_action(ControllerAction.RETRIEVE, "c1", None)

        # Second retrieval for same claim must be rejected
        can_afford2, reason = tracker.can_afford(ControllerAction.RETRIEVE, "c1")
        self.assertFalse(can_afford2)
        self.assertIn("max retrievals", reason)


class TestCostAggregationAndUnknowns(unittest.TestCase):
    """Test accurate aggregation of multi-dimensional costs including None fields."""

    def test_cost_aggregation_with_none_fields(self):
        tracker = BudgetTracker(VerificationBudget(max_steps=10))

        cost1 = VerificationCost(model_calls=1, prompt_tokens=100, completion_tokens=50, latency_seconds=0.5)
        cost2 = VerificationCost(model_calls=0, prompt_tokens=None, completion_tokens=None, latency_seconds=0.2)

        tracker.record_action(ControllerAction.VERIFY, "c1", cost1)
        tracker.record_action(ControllerAction.RETRIEVE, "c1", cost2)

        global_cost = tracker.get_global_cost()
        self.assertEqual(global_cost["steps_taken"], 2)
        self.assertEqual(global_cost["model_calls"], 1)
        self.assertEqual(global_cost["input_tokens"], 100)
        self.assertEqual(global_cost["output_tokens"], 50)
        self.assertEqual(global_cost["total_tokens"], 150)
        self.assertAlmostEqual(global_cost["latency_seconds"], 0.7, places=2)


class TestRobustnessAndErrorHandling(unittest.TestCase):
    """Test controller resilience against edge cases, exceptions, and invalid inputs."""

    def test_empty_claims_list(self):
        controller = VerificationController()
        trace = controller.run([])
        self.assertEqual(trace.total_steps, 0)
        self.assertEqual(trace.stopping_reason, StoppingReason.NO_ELIGIBLE_ACTIONS)

    def test_duplicate_claim_ids_raises_value_error(self):
        c1 = _make_claim("dup_id", "Claim 1")
        c2 = _make_claim("dup_id", "Claim 2")
        controller = VerificationController()
        with self.assertRaises(ValueError) as ctx:
            controller.run([c1, c2])
        self.assertIn("Duplicate claim ID", str(ctx.exception))

    def test_action_exception_handled_gracefully(self):
        claim = _make_claim("c1", "Claim text")

        class FaultyRetriever(BaseRetriever):
            def retrieve(self, claim, top_k=3):
                raise ConnectionError("Qdrant socket timeout simulated")

        controller = VerificationController(
            retriever=FaultyRetriever(),
            global_budget=VerificationBudget(max_steps=5),
        )

        trace = controller.run([claim])
        # Controller should not crash; claim transitions to UNRESOLVED
        self.assertEqual(claim.state, ClaimState.UNRESOLVED)
        error_steps = [s for s in trace.steps if "FAILED" in (s.action_outcome or "")]
        self.assertTrue(len(error_steps) > 0)

    def test_assessor_technical_error_handling(self):
        """When assessor returns is_error=True, claim transitions to UNRESOLVED with technical error."""
        from src.controller.reward import ProxyVerificationRewardModel

        claim = _make_claim("c_tech_err", "Some claim text")
        passage = _make_passage("ev1", "Doc", "Some text")

        class BrokenAssessor(BaseAssessor):
            @property
            def assessor_id(self) -> str:
                return "broken_assessor"

            def assess(self, claim, evidence):
                return EvidenceAssessment(
                    claim_id=claim.claim_id,
                    evidence_ids=["ev1"],
                    supporting_spans={},
                    label=AssessmentLabel.INSUFFICIENT,
                    confidence=0.0,
                    rationale="LLM call failed: RuntimeError: Context window exceeded",
                    is_error=True,
                    error_type="RuntimeError",
                )

        controller = VerificationController(
            retriever=MockRetriever(canned_evidence={"c_tech_err": [passage]}),
            assessor=BrokenAssessor(),
            global_budget=VerificationBudget(max_steps=5),
        )

        trace = controller.run([claim])
        self.assertEqual(claim.state, ClaimState.UNRESOLVED)
        verify_step = [s for s in trace.steps if s.action == ControllerAction.VERIFY][0]
        self.assertIsNotNone(verify_step.error)
        self.assertIn("TECHNICAL ERROR: RuntimeError", verify_step.action_outcome)

        # Verify reward model treats this as failure
        reward_model = ProxyVerificationRewardModel()
        outcome = reward_model.compute_reward(
            claim=claim,
            action=ControllerAction.VERIFY,
            assessment=EvidenceAssessment(
                claim_id="c_tech_err",
                evidence_ids=["ev1"],
                supporting_spans={},
                label=AssessmentLabel.INSUFFICIENT,
                confidence=0.0,
                rationale="LLM call failed: RuntimeError",
                is_error=True,
                error_type="RuntimeError",
            ),
        )
        self.assertTrue(outcome.is_action_failure)
        self.assertEqual(outcome.quality_reward, -0.8)

    def test_deterministic_tie_breaking(self):
        # Two identical claims with identical CTS and type
        c_b = _make_claim("claim_b", "Identical proposition", triage_score=0.7)
        c_a = _make_claim("claim_a", "Identical proposition", triage_score=0.7)

        policy = SequentialDecisionPolicy()
        budget = BudgetTracker(VerificationBudget(max_steps=10))

        # Irrespective of input list order, claim_a must be picked first by ID
        _, selected1, _ = policy.select_next_action([c_b, c_a], {}, {}, {}, budget)
        self.assertEqual(selected1.claim_id, "claim_a")

        _, selected2, _ = policy.select_next_action([c_a, c_b], {}, {}, {}, budget)
        self.assertEqual(selected2.claim_id, "claim_a")


class TestLifecycleInvariants(unittest.TestCase):
    """Ensure state machine rules and separation of assessment from verification."""

    def test_supported_assessment_below_threshold_does_not_verify(self):
        claim = _make_claim("c1", "Claim text")
        passage = _make_passage("ev1", "Doc", "Claim text")

        # Mock assessor that returns SUPPORTED with low confidence (0.4)
        class LowConfidenceAssessor(BaseAssessor):
            @property
            def assessor_id(self) -> str:
                return "low_conf"

            def assess(self, claim, evidence):
                return EvidenceAssessment(
                    claim_id=claim.claim_id,
                    evidence_ids=["ev1"],
                    supporting_spans={},
                    label=AssessmentLabel.SUPPORTED,
                    confidence=0.40,  # Below default 0.60 threshold
                    rationale="Weak support",
                    limitations="Low confidence",
                )

        controller = VerificationController(
            retriever=MockRetriever(canned_evidence={"c1": [passage]}),
            assessor=LowConfidenceAssessor(),
            global_budget=VerificationBudget(max_steps=5),
            policy=SequentialDecisionPolicy(ControllerPolicyConfig(verification_confidence_threshold=0.60)),
        )

        trace = controller.run([claim])
        # Low confidence support must NOT transition to VERIFIED
        self.assertNotEqual(claim.state, ClaimState.VERIFIED)


class TestTraceSerialization(unittest.TestCase):
    """Ensure complete, valid JSON serialization of the controller trace."""

    def test_trace_json_roundtrip(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        passage = _make_passage("ev1", "Doc", "Arthur's Magazine was published in Philadelphia.")

        controller = VerificationController(
            retriever=MockRetriever(canned_evidence={"c1": [passage]}),
            assessor=MockDeterministicAssessor(),
            global_budget=VerificationBudget(max_steps=5),
        )

        trace = controller.run([claim])
        json_str = trace.to_json()
        parsed = json.loads(json_str)

        self.assertEqual(parsed["run_id"], trace.run_id)
        self.assertEqual(parsed["total_steps"], trace.total_steps)
        self.assertIn("steps", parsed)
        self.assertTrue(len(parsed["steps"]) > 0)
        self.assertIn("claims_summary", parsed)


class TestAdditionalControllerInvariants(unittest.TestCase):
    """Further tests for budget edge cases, state history integrity, and token limits."""

    def test_zero_budget_stops_cleanly(self):
        c1 = _make_claim("c1", "Some claim")
        controller = VerificationController(
            global_budget=VerificationBudget(max_steps=0),
        )
        trace = controller.run([c1])
        self.assertEqual(trace.stopping_reason, StoppingReason.BUDGET_EXHAUSTED)
        self.assertNotEqual(c1.state, ClaimState.VERIFIED)

    def test_max_steps_hard_limit_enforced(self):
        # 3 claims that all need retrieval and verification (would take 6 steps)
        claims = [_make_claim(f"c{i}", f"Claim {i}") for i in range(3)]
        controller = VerificationController(
            retriever=MockRetriever(),
            assessor=MockDeterministicAssessor(),
            global_budget=VerificationBudget(max_steps=3),
        )
        trace = controller.run(claims)
        self.assertEqual(trace.stopping_reason, StoppingReason.MAX_STEPS_REACHED)
        self.assertLessEqual(trace.total_steps, 3)

    def test_global_token_budget_exhaustion(self):
        c1 = _make_claim("c1", "Claim text")
        passage = _make_passage("ev1", "Doc", "Claim text")

        class TokenAssessor(BaseAssessor):
            @property
            def assessor_id(self) -> str:
                return "token_assessor"

            def assess(self, claim, evidence):
                res = MockDeterministicAssessor().assess(claim, evidence)
                res.cost = VerificationCost(model_calls=1, total_tokens=500, latency_seconds=0.01)
                return res

        controller = VerificationController(
            retriever=MockRetriever(canned_evidence={"c1": [passage]}),
            assessor=TokenAssessor(),
            global_budget=VerificationBudget(max_total_tokens=400, max_steps=10),
        )

        trace = controller.run([c1, _make_claim("c2", "Claim 2")])
        self.assertEqual(trace.stopping_reason, StoppingReason.BUDGET_EXHAUSTED)
        self.assertGreaterEqual(trace.total_cost["total_tokens"], 500)

    def test_per_claim_verification_budget_enforcement(self):
        tracker = BudgetTracker(
            global_budget=VerificationBudget(max_steps=20),
            per_claim_budget=PerClaimBudget(max_verifications=1),
        )
        can_v1, _ = tracker.can_afford(ControllerAction.VERIFY, "c1")
        self.assertTrue(can_v1)
        tracker.record_action(ControllerAction.VERIFY, "c1", None)

        can_v2, reason = tracker.can_afford(ControllerAction.VERIFY, "c1")
        self.assertFalse(can_v2)
        self.assertIn("max verifications", reason)

    def test_state_history_contains_only_valid_transitions(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        passage = _make_passage("ev1", "Doc", "Arthur's Magazine was published in Philadelphia.")

        controller = VerificationController(
            retriever=MockRetriever(canned_evidence={"c1": [passage]}),
            assessor=MockDeterministicAssessor(),
            global_budget=VerificationBudget(max_steps=10),
        )

        trace = controller.run([claim])
        self.assertEqual(claim.state, ClaimState.VERIFIED)

        # In state_history:
        # Initial: PENDING
        # Step: PENDING -> SELECTED
        # Step: SELECTED -> VERIFIED
        states = [h["to_state"] for h in claim.state_history]
        self.assertEqual(states, [ClaimState.PENDING.value, ClaimState.SELECTED.value, ClaimState.VERIFIED.value])

    def test_repair_records_rationale_and_evidence_ids(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        contra_passage = _make_passage(
            "ev_contra", "Arthur's Magazine",
            "Arthur's Magazine was not published in Philadelphia. It was based in Boston."
        )

        repairer = MockRepairer()
        controller = VerificationController(
            retriever=MockRetriever(canned_evidence={"c1": [contra_passage]}),
            assessor=MockDeterministicAssessor(),
            repairer=repairer,
            global_budget=VerificationBudget(max_steps=10),
        )

        trace = controller.run([claim])
        repair_steps = [s for s in trace.steps if s.action == ControllerAction.REPAIR.value]
        self.assertEqual(len(repair_steps), 1)
        self.assertIn("ev_contra", repair_steps[0].action_outcome)
        self.assertIsNotNone(repair_steps[0].selection_rationale)

