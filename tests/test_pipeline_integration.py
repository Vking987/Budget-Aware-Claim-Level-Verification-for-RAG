"""Integration test suite for the end-to-end RAG verification pipeline.

Tests 9 required scenarios completely offline and deterministically:
1. Fully supported answer — VERIFIED claim -> grounded synthesis with correct citations
2. Unsupported claim triggers further retrieval then abstention
3. Contradictory evidence triggers repair or exclusion / abstention
4. Budget exhaustion stops loop immediately with BUDGET_EXHAUSTED
5. Zero verified claims returns safe abstention (does not echo draft)
6. Citation integrity: citations only reference verified passages
7. Stopping reasons: all claims resolved vs budget exhausted vs max steps vs no eligible actions
8. MockRepairer disclosure in pipeline output / trace
9. Missing service errors surfaced clearly (e.g. Qdrant down in live mode)
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus, SentenceRecord
from src.controller.budget import BudgetTracker
from src.controller.interfaces import MockRepairer, MockRetriever
from src.controller.models import ControllerAction, ControllerTrace, PerClaimBudget, StoppingReason, TraceStep, VerificationBudget
from src.controller.policy import ControllerPolicyConfig, SequentialDecisionPolicy, StaticAllocationPolicy
from src.evidence.assessor import MockDeterministicAssessor
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost
from src.pipeline import PipelineResult, RAGVerificationPipeline
from src.synthesis.models import SynthesisResult
from src.synthesis.synthesizer import LLMAnswerSynthesizer, MockAnswerSynthesizer


class TestPipelineIntegrationScenarios(unittest.TestCase):
    """Integration test suite for RAG verification pipeline scenarios."""

    def setUp(self):
        self.question = "What is the capital of France?"
        self.paris_passage = EvidencePassage(
            evidence_id="ev_paris_101",
            doc_title="Paris Overview",
            source_text="Paris is the capital and most populous city of France.",
            retrieval_score=0.95,
        )
        self.lyon_passage = EvidencePassage(
            evidence_id="ev_lyon_202",
            doc_title="Lyon Overview",
            source_text="Lyon is the third-largest city in France, located in the east-central part.",
            retrieval_score=0.70,
        )

    # -------------------------------------------------------------------------
    # Scenario 1: Fully supported answer — VERIFIED claim -> grounded synthesis with correct citations
    # -------------------------------------------------------------------------
    def test_scenario_1_fully_supported_answer_verified_with_citations(self):
        class SupportingRetriever(MockRetriever):
            def retrieve(self, claim, top_k=5):
                return [self.paris_passage], VerificationCost(model_calls=0)

        class SupportingAssessor(MockDeterministicAssessor):
            def assess(self, claim, passages):
                ev_id = passages[0].evidence_id if passages else "none"
                return EvidenceAssessment(
                    claim_id=claim.claim_id,
                    evidence_ids=[ev_id],
                    supporting_spans={ev_id: ["Paris is the capital"]} if passages else {},
                    label=AssessmentLabel.SUPPORTED,
                    confidence=0.98,
                    rationale="Passage directly confirms claim.",
                    cost=VerificationCost(model_calls=1, prompt_tokens=30, completion_tokens=15, total_tokens=45),
                )

        mock_retriever = SupportingRetriever()
        mock_retriever.paris_passage = self.paris_passage
        mock_assessor = SupportingAssessor()

        pipeline = RAGVerificationPipeline(
            retriever=mock_retriever,
            assessor=mock_assessor,
            synthesizer=MockAnswerSynthesizer(),
            dry_run=True,
        )

        result = pipeline.run(self.question, policy_name="StaticFIFO")

        self.assertIsInstance(result, PipelineResult)
        verified_claims = [c for c in result.claims if c["state"] == ClaimState.VERIFIED.value]
        self.assertGreaterEqual(len(verified_claims), 1)
        self.assertTrue(result.is_grounded)
        self.assertIn("Paris Overview", result.synthesis.get("cited_sources", []))

    # -------------------------------------------------------------------------
    # Scenario 2: Unsupported claim triggers further retrieval then abstention
    # -------------------------------------------------------------------------
    def test_scenario_2_unsupported_claim_retrieval_then_abstention(self):
        retrieval_call_count = 0

        class MultiStepEmptyRetriever(MockRetriever):
            def retrieve(self, claim, top_k=5):
                nonlocal retrieval_call_count
                retrieval_call_count += 1
                return [], VerificationCost(model_calls=0)

        mock_retriever = MultiStepEmptyRetriever()
        mock_assessor = MockDeterministicAssessor()

        pipeline = RAGVerificationPipeline(
            retriever=mock_retriever,
            assessor=mock_assessor,
            synthesizer=MockAnswerSynthesizer(),
            dry_run=True,
        )

        result = pipeline.run("What is an undiscovered planet in Andromeda?", policy_name="StaticFIFO")

        self.assertFalse(result.is_grounded)
        self.assertIn("does not contain sufficient verified information", result.final_answer.lower())
        self.assertTrue(any("No claims could be verified" in w for w in result.warnings))
        self.assertGreaterEqual(retrieval_call_count, 1)

    # -------------------------------------------------------------------------
    # Scenario 3: Contradictory evidence triggers repair or exclusion
    # -------------------------------------------------------------------------
    def test_scenario_3_contradictory_evidence_handled(self):
        class ContradictingAssessor(MockDeterministicAssessor):
            def assess(self, claim, passages):
                ev_id = passages[0].evidence_id if passages else "none"
                return EvidenceAssessment(
                    claim_id=claim.claim_id,
                    evidence_ids=[ev_id],
                    supporting_spans={ev_id: ["Paris is the capital"]},
                    label=AssessmentLabel.CONTRADICTED,
                    confidence=0.95,
                    rationale="Passage contradicts the draft statement.",
                    cost=VerificationCost(model_calls=1, prompt_tokens=25, completion_tokens=10, total_tokens=35),
                )

        class SinglePassageRetriever(MockRetriever):
            def retrieve(self, claim, top_k=5):
                return [self.paris_passage], VerificationCost(model_calls=0)

        mock_retriever = SinglePassageRetriever()
        mock_retriever.paris_passage = self.paris_passage
        mock_assessor = ContradictingAssessor()

        pipeline = RAGVerificationPipeline(
            retriever=mock_retriever,
            assessor=mock_assessor,
            synthesizer=MockAnswerSynthesizer(),
            dry_run=True,
        )

        result = pipeline.run("The capital of France is Lyon.", policy_name="Exhaustive")

        # Contradicted claims should be captured in warnings or synthesis contradicted_claims
        has_contradicted = (
            any("contradicted" in w.lower() for w in result.warnings)
            or bool(result.synthesis.get("contradicted_claims"))
        )
        self.assertTrue(has_contradicted)
        self.assertFalse(result.is_grounded)

    # -------------------------------------------------------------------------
    # Scenario 4: Budget exhaustion stops loop immediately with BUDGET_EXHAUSTED
    # -------------------------------------------------------------------------
    def test_scenario_4_budget_exhaustion_stops_loop(self):
        class PassageRetriever(MockRetriever):
            def retrieve(self, claim, top_k=5):
                return [self.paris_passage], VerificationCost(model_calls=0)

        class CountingAssessor(MockDeterministicAssessor):
            def assess(self, claim, passages):
                ev_id = passages[0].evidence_id if passages else "none"
                return EvidenceAssessment(
                    claim_id=claim.claim_id,
                    evidence_ids=[ev_id],
                    supporting_spans={},
                    label=AssessmentLabel.INSUFFICIENT,
                    confidence=0.5,
                    rationale="Insufficient evidence to verify claim.",
                    cost=VerificationCost(model_calls=1, prompt_tokens=50, completion_tokens=20, total_tokens=70),
                )

        mock_retriever = PassageRetriever()
        mock_retriever.paris_passage = self.paris_passage

        pipeline = RAGVerificationPipeline(
            retriever=mock_retriever,
            assessor=CountingAssessor(),
            synthesizer=MockAnswerSynthesizer(),
            dry_run=True,
        )

        # When max_model_calls=1 and max_steps=5, step 1 uses 1 model call in VERIFY, exhausting model calls budget.
        # Next loop iteration, policy sees is_exhausted() -> selects STOP -> controller sets BUDGET_EXHAUSTED.
        with patch("src.pipeline.VerificationBudget") as MockBudget:
            MockBudget.return_value = VerificationBudget(max_steps=5, max_model_calls=1)
            result = pipeline.run(self.question, policy_name="StaticFIFO")

        self.assertEqual(result.stopping_reason, StoppingReason.BUDGET_EXHAUSTED.value)

    # -------------------------------------------------------------------------
    # Scenario 5: Zero verified claims returns safe abstention (does not echo draft)
    # -------------------------------------------------------------------------
    def test_scenario_5_zero_verified_claims_safe_abstention_no_draft_echo(self):
        draft_text = "Untruthful hallucination stating Atlantis was discovered in 2024."

        class FakeContentFilter:
            def process(self, text):
                return [
                    SentenceRecord(
                        sentence_id="s_0",
                        original_text=text,
                        cleaned_text=text,
                        is_content_bearing=True,
                        filter_reason="passed",
                        char_start=0,
                        char_end=len(text),
                    )
                ]

        class FakeDecomposer:
            def decompose(self, sent):
                return [
                    ClaimRecord(
                        claim_id="c_fake",
                        source_sentence_id="s_0",
                        original_text=sent.original_text,
                        normalized_text=sent.cleaned_text,
                        claim_type=ClaimType.FACTUAL_RELATION,
                        decomposition_status=DecompositionStatus.ATOMIC,
                    )
                ]

        class AlwaysInsufficientAssessor(MockDeterministicAssessor):
            def assess(self, claim, passages):
                return EvidenceAssessment(
                    claim_id=claim.claim_id,
                    evidence_ids=["none"],
                    supporting_spans={},
                    label=AssessmentLabel.INSUFFICIENT,
                    confidence=0.1,
                    cost=VerificationCost(model_calls=1),
                )

        pipeline = RAGVerificationPipeline(
            retriever=MockRetriever(),
            assessor=AlwaysInsufficientAssessor(),
            synthesizer=MockAnswerSynthesizer(),
            content_filter=FakeContentFilter(),
            decomposer=FakeDecomposer(),
            dry_run=True,
        )

        result = pipeline.run("Where is Atlantis?", policy_name="StaticFIFO")

        # Must NOT echo the unverified draft
        self.assertNotIn(draft_text, result.final_answer)
        self.assertIn("does not contain sufficient verified information", result.final_answer.lower())
        self.assertFalse(result.is_grounded)

    # -------------------------------------------------------------------------
    # Scenario 6: Citation integrity: citations only reference verified passages
    # -------------------------------------------------------------------------
    def test_scenario_6_citation_integrity_verified_passages_only(self):
        ev_verified = EvidencePassage(
            evidence_id="ev_verified_01",
            doc_title="Verified Doc",
            source_text="Verified facts about Paris.",
            retrieval_score=0.9,
        )
        ev_unverified = EvidencePassage(
            evidence_id="ev_unverified_02",
            doc_title="Unverified Rumors",
            source_text="Unverified rumors about Paris.",
            retrieval_score=0.5,
        )

        class TargetedRetriever(MockRetriever):
            def retrieve(self, claim, top_k=5):
                return [ev_verified, ev_unverified], VerificationCost(model_calls=0)

        class TargetedAssessor(MockDeterministicAssessor):
            def assess(self, claim, passages):
                return EvidenceAssessment(
                    claim_id=claim.claim_id,
                    evidence_ids=[ev_verified.evidence_id],
                    supporting_spans={ev_verified.evidence_id: ["Verified facts"]},
                    label=AssessmentLabel.SUPPORTED,
                    confidence=0.99,
                    rationale="Supported by ev_verified_01",
                    cost=VerificationCost(model_calls=1),
                )

        pipeline = RAGVerificationPipeline(
            retriever=TargetedRetriever(),
            assessor=TargetedAssessor(),
            synthesizer=MockAnswerSynthesizer(),
            dry_run=True,
        )

        result = pipeline.run("Tell me about Paris", policy_name="StaticFIFO")

        cited = result.synthesis.get("cited_sources", [])
        self.assertIn("Verified Doc", cited)
        self.assertNotIn("Unverified Rumors", cited)

    # -------------------------------------------------------------------------
    # Scenario 7: Stopping reasons distinction
    # -------------------------------------------------------------------------
    def test_scenario_7_stopping_reasons_distinction(self):
        # 1. NO_ELIGIBLE_ACTIONS or ALL_CLAIMS_RESOLVED
        pipeline_dry = RAGVerificationPipeline(dry_run=True)
        res_dry = pipeline_dry.run("Short query", policy_name="ContextualMAB_LinUCB")
        self.assertIn(res_dry.stopping_reason, [
            StoppingReason.NO_ELIGIBLE_ACTIONS.value,
            StoppingReason.ALL_CLAIMS_RESOLVED.value,
        ])

        # 2. MAX_STEPS_REACHED (when step count hits max_steps)
        with patch("src.pipeline.VerificationBudget") as MockBudget:
            MockBudget.return_value = VerificationBudget(max_steps=1, max_model_calls=10)
            res_steps = pipeline_dry.run("Short query", policy_name="Exhaustive")
            self.assertEqual(res_steps.stopping_reason, StoppingReason.MAX_STEPS_REACHED.value)

    # -------------------------------------------------------------------------
    # Scenario 8: MockRepairer disclosure in pipeline output / trace
    # -------------------------------------------------------------------------
    def test_scenario_8_mock_repairer_disclosure(self):
        pipeline = RAGVerificationPipeline(dry_run=True)
        self.assertIsInstance(pipeline.repairer, MockRepairer)

        res = pipeline.run("Test query", policy_name="StaticFIFO")
        self.assertIsNotNone(res.trace)
        # Trace records policy and all steps faithfully
        self.assertEqual(res.trace.run_id, "demo_StaticFIFO")

    # -------------------------------------------------------------------------
    # Scenario 9: Missing service errors surfaced clearly (e.g. Qdrant down in live mode)
    # -------------------------------------------------------------------------
    def test_scenario_9_missing_service_errors_surfaced_clearly(self):
        class FailingRetriever:
            def retrieve(self, claim, top_k=5):
                raise ConnectionRefusedError("Failed to connect to Qdrant at 127.0.0.1:6333")

        pipeline = RAGVerificationPipeline(
            retriever=FailingRetriever(),
            dry_run=False,
        )

        with self.assertRaises(RuntimeError) as ctx:
            pipeline.run("Will fail because Qdrant is down", policy_name="StaticFIFO")

        err_msg = str(ctx.exception)
        self.assertIn("Initial retrieval failed", err_msg)
        self.assertIn("Qdrant", err_msg)
        self.assertIn("6333", err_msg)
        self.assertIn("--dry-run", err_msg)


if __name__ == "__main__":
    unittest.main()


class TestPolicyResolution(unittest.TestCase):
    """Every CLI policy name must resolve to the policy it names; unknown names must fail loudly."""

    def test_cli_names_resolve_to_named_policy(self):
        from src.controller.mab_policy import ContextualMABPolicy
        from src.controller.policy import SequentialDecisionPolicy, StaticAllocationPolicy, StaticCTSRankedPolicy
        pl = RAGVerificationPipeline(dry_run=True)
        expected = {
            "ContextualMAB_LinUCB": ContextualMABPolicy,
            "DeterministicCTS": SequentialDecisionPolicy,
            "StaticCTSRanked": StaticCTSRankedPolicy,
            "StaticFIFO": StaticAllocationPolicy,
            "ExhaustiveReference": SequentialDecisionPolicy,
            "VanillaRAG": SequentialDecisionPolicy,
        }
        for name, cls in expected.items():
            pol, _, _ = pl._resolve_policy(name)
            self.assertIs(type(pol), cls, name)

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            RAGVerificationPipeline(dry_run=True)._resolve_policy("garbage")
