"""Unit and integration tests for End-to-End Pipeline and Answer Synthesis (Stage E).

Tests:
1. End-to-end question processing with mocked dependencies.
2. Final synthesis using supported claims.
3. Exclusion of contradicted claims from affirmative answers.
4. Safe handling of unresolved or insufficient evidence.
5. Citation/evidence traceability.
6. Synthesis failures and malformed model responses.
7. Complete resource-cost accounting.
8. Real-backend smoke test (executable separately or when services are live).
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus
from src.controller.budget import BudgetTracker, VerificationBudget
from src.controller.interfaces import LLMRepairer, MockRepairer, MockRetriever
from src.controller.models import ControllerAction, ControllerTrace, StoppingReason, TraceStep
from src.evidence.assessor import MockDeterministicAssessor
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost
from src.pipeline import PipelineResult, RAGVerificationPipeline
from src.synthesis.models import SynthesisResult
from src.synthesis.synthesizer import LLMAnswerSynthesizer, MockAnswerSynthesizer


class TestEndToEndAnswerSynthesis(unittest.TestCase):
    """Test suite for answer synthesis and pipeline grounding."""

    def setUp(self):
        self.question = "What is the capital of France?"
        self.draft_answer = "The capital of France is Paris."
        self.passages = [
            EvidencePassage(
                evidence_id="ev_paris_1",
                doc_title="Paris Overview",
                source_text="Paris is the capital and most populous city of France.",
                retrieval_score=0.92,
            ),
            EvidencePassage(
                evidence_id="ev_lyon_1",
                doc_title="Lyon Overview",
                source_text="Lyon is the third-largest city in France, located in the east-central part.",
                retrieval_score=0.75,
            ),
        ]
        self.evidence_store = {"c1": self.passages}

    # 1. End-to-end question processing with mocked dependencies
    def test_end_to_end_pipeline_mock(self):
        pipeline = RAGVerificationPipeline(dry_run=True)
        res = pipeline.run("What is the capital of France?", policy_name="StaticFIFO")
        self.assertIsInstance(res, PipelineResult)
        self.assertEqual(res.question, "What is the capital of France?")
        self.assertIsNotNone(res.final_answer)
        self.assertGreater(len(res.final_answer), 0)
        self.assertIn("model_calls", res.total_cost)
        self.assertIn("latency_seconds", res.total_cost)

    # 2. Final synthesis using supported claims
    def test_synthesis_uses_supported_claims(self):
        claim_verified = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Paris is the capital of France.",
            normalized_text="Paris is the capital of France.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
        )
        synth = MockAnswerSynthesizer()
        res = synth.synthesize(
            question=self.question,
            initial_draft_answer=self.draft_answer,
            claims=[claim_verified],
            evidence_store=self.evidence_store,
        )
        self.assertTrue(res.is_grounded)
        self.assertIn("Paris is the capital of France.", res.final_answer)
        self.assertIn("Paris Overview", res.cited_sources)
        self.assertFalse(res.has_unresolved_limitations)

    # 3. Exclusion of contradicted claims from affirmative answers
    def test_synthesis_excludes_contradicted_claims(self):
        claim_contradicted = ClaimRecord(
            claim_id="c2",
            source_sentence_id="s1",
            original_text="Lyon is the capital of France.",
            normalized_text="Lyon is the capital of France.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.UNRESOLVED,
        )
        assmt = EvidenceAssessment(
            claim_id="c2",
            evidence_ids=["ev_paris_1"],
            supporting_spans={},
            label=AssessmentLabel.CONTRADICTED,
            confidence=0.85,
            rationale="Evidence states Paris is the capital, contradicting Lyon.",
        )
        synth = MockAnswerSynthesizer()
        res = synth.synthesize(
            question="Is Lyon the capital of France?",
            initial_draft_answer="Lyon is the capital of France.",
            claims=[claim_contradicted],
            evidence_store={"c2": self.passages},
            assessments={"c2": assmt},
        )
        self.assertTrue(any("Lyon" in c for c in res.contradicted_claims))
        self.assertNotIn("According to [Paris Overview], Lyon is the capital of France", res.final_answer)
        self.assertTrue(res.has_unresolved_limitations)

    # 4. Safe handling of unresolved or insufficient evidence
    def test_synthesis_safe_handling_unresolved(self):
        claim_unresolved = ClaimRecord(
            claim_id="c3",
            source_sentence_id="s1",
            original_text="The city has 42 bridges.",
            normalized_text="The city has 42 bridges.",
            claim_type=ClaimType.NUMERICAL_QUANTITATIVE,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.UNRESOLVED,
        )
        assmt = EvidenceAssessment(
            claim_id="c3",
            evidence_ids=[],
            supporting_spans={},
            label=AssessmentLabel.INSUFFICIENT,
            confidence=0.20,
            rationale="Passages do not mention bridges.",
        )
        synth = MockAnswerSynthesizer()
        res = synth.synthesize(
            question="How many bridges does Paris have?",
            initial_draft_answer="The city has 42 bridges.",
            claims=[claim_unresolved],
            evidence_store={"c3": self.passages},
            assessments={"c3": assmt},
        )
        self.assertFalse(res.is_grounded)
        self.assertTrue("insufficient" in res.final_answer.lower() or "not contain sufficient" in res.final_answer.lower())
        self.assertTrue(res.has_unresolved_limitations)

    # 5. Citation and evidence traceability
    def test_citation_evidence_traceability(self):
        claim_verified = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Paris is the capital of France.",
            normalized_text="Paris is the capital of France.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
        )
        synth = MockAnswerSynthesizer()
        res = synth.synthesize(
            question=self.question,
            initial_draft_answer=self.draft_answer,
            claims=[claim_verified],
            evidence_store=self.evidence_store,
        )
        for cited_title in res.cited_sources:
            all_titles = [p.doc_title for p_list in self.evidence_store.values() for p in p_list]
            self.assertIn(cited_title, all_titles)

    # 6. Synthesis failures and malformed model responses
    def test_synthesis_model_failure_fallback(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Local LLM timeout")
        mock_client.models.list.return_value = MagicMock(data=[MagicMock(id="test_model")])

        claim_verified = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Paris is the capital of France.",
            normalized_text="Paris is the capital of France.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
        )
        synth = LLMAnswerSynthesizer(client=mock_client)
        res = synth.synthesize(
            question=self.question,
            initial_draft_answer=self.draft_answer,
            claims=[claim_verified],
            evidence_store=self.evidence_store,
        )
        self.assertEqual(res.synthesis_method, "fallback_grounded")
        self.assertIsNotNone(res.error)
        # the deterministic stitched fallback restates the verified claim, with its citation before the final period
        self.assertIn("Paris is the capital of France [Paris Overview].", res.final_answer)
        self.assertEqual(res.answer_violations, [])

    # 7. Complete resource-cost accounting
    def test_complete_resource_cost_accounting(self):
        mock_retriever = MockRetriever(
            default_cost=VerificationCost(model_calls=1, prompt_tokens=10, completion_tokens=5, total_tokens=15, latency_seconds=0.1)
        )
        mock_synth = MockAnswerSynthesizer(
            default_cost=VerificationCost(model_calls=1, prompt_tokens=20, completion_tokens=10, total_tokens=30, latency_seconds=0.2)
        )
        pipeline = RAGVerificationPipeline(
            retriever=mock_retriever,
            synthesizer=mock_synth,
            dry_run=True,
        )
        res = pipeline.run("What is Paris?", policy_name="StaticFIFO")
        tot = res.total_cost
        self.assertIn("model_calls", tot)
        self.assertIn("total_tokens", tot)
        self.assertIn("latency_seconds", tot)
        self.assertGreaterEqual(tot["model_calls"], 1)

    # 8. Real-backend smoke test (can be executed when live services are reachable)
    def test_real_backend_smoke_test(self):
        # Skip gracefully if optional ML/retrieval dependencies are not installed.
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            self.skipTest("sentence_transformers not installed; skipping live smoke test.")
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            self.skipTest("qdrant_client not installed; skipping live smoke test.")

        import requests
        try:
            r = requests.get("http://127.0.0.1:8080/health", timeout=1.0)
            server_online = (r.status_code == 200)
        except Exception:
            server_online = False

        if not server_online:
            self.skipTest("Local llama-server offline; skipping live smoke test.")

        # Also verify Qdrant is reachable before attempting retrieval.
        try:
            rq = requests.get("http://127.0.0.1:6333/healthz", timeout=1.0)
            qdrant_online = (rq.status_code == 200)
        except Exception:
            qdrant_online = False

        if not qdrant_online:
            self.skipTest("Local Qdrant offline; skipping live smoke test.")

        try:
            pipeline = RAGVerificationPipeline(dry_run=False)
            test_q = "What State has a Disney Resort & Spa?"
            res = pipeline.run(test_q, policy_name="ContextualMAB_LinUCB")
        except OSError as e:
            if "paging file" in str(e).lower() or "1455" in str(e):
                self.skipTest(f"Insufficient paging file/system memory to load embedding model: {e}")
            raise
        self.assertIsInstance(res, PipelineResult)
        self.assertGreater(len(res.final_answer), 0)
        self.assertIn("model_calls", res.total_cost)
        self.assertGreaterEqual(res.total_cost["model_calls"], 1)


    # 9. Claim stitching: 3 decomposed claims, original order, supported/contradicted/repaired, citations
    def test_three_decomposed_claims_stitching_order_repaired_provenance(self):
        """Task 3 verification: Verify claim stitching across multiple claims in original order.
        
        Three decomposed claims:
        - c_0: SUPPORTED / VERIFIED (factual fact with evidence provenance)
        - c_1: CONTRADICTED / UNRESOLVED (false draft fact)
        - c_2: SUPPORTED / VERIFIED (factual fact with evidence provenance)
        
        Verifies:
        - Final synthesized answer is coherent and preserves the order of verified facts.
        - Uses repaired wording if/when a repaired claim is provided.
        - Contradicted/unsupported claim is NOT presented as fact.
        - Preserves evidence provenance / citations for verified claims.
        """
        c0 = ClaimRecord(
            claim_id="c_0",
            source_sentence_id="s_0",
            original_text="Paris is the capital of France.",
            normalized_text="Paris is the capital of France.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
            evidence_provenance={"supporting_spans": {"ev_paris_1": ["capital and most populous city"]}},
        )
        c1 = ClaimRecord(
            claim_id="c_1",
            source_sentence_id="s_1",
            original_text="Lyon is the largest city in France.",
            normalized_text="Lyon is the largest city in France.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.UNRESOLVED,
        )
        c2 = ClaimRecord(
            claim_id="c_2",
            source_sentence_id="s_2",
            original_text="Lyon is the third-largest city in France.",
            normalized_text="Lyon is the third-largest city in France.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
            evidence_provenance={"supporting_spans": {"ev_lyon_1": ["third-largest city in France"]}},
        )

        assmts = {
            "c_0": EvidenceAssessment(
                claim_id="c_0",
                evidence_ids=["ev_paris_1"],
                supporting_spans={"ev_paris_1": ["capital and most populous city"]},
                label=AssessmentLabel.SUPPORTED,
                confidence=0.95,
                rationale="Directly confirmed by Paris Overview.",
            ),
            "c_1": EvidenceAssessment(
                claim_id="c_1",
                evidence_ids=["ev_paris_1", "ev_lyon_1"],
                supporting_spans={},
                label=AssessmentLabel.CONTRADICTED,
                confidence=0.90,
                rationale="Paris is most populous, Lyon is third-largest.",
            ),
            "c_2": EvidenceAssessment(
                claim_id="c_2",
                evidence_ids=["ev_lyon_1"],
                supporting_spans={"ev_lyon_1": ["third-largest city in France"]},
                label=AssessmentLabel.SUPPORTED,
                confidence=0.95,
                rationale="Directly confirmed by Lyon Overview.",
            ),
        }

        evidence_store = {
            "c_0": [self.passages[0]],  # Paris Overview
            "c_1": self.passages,
            "c_2": [self.passages[1]],  # Lyon Overview
        }

        synth = MockAnswerSynthesizer()
        res = synth.synthesize(
            question="What are the major cities of France?",
            initial_draft_answer="Paris is the capital of France. Lyon is the largest city in France.",
            claims=[c0, c1, c2],
            evidence_store=evidence_store,
            assessments=assmts,
        )

        # 1. Answer is coherent and grounded
        self.assertTrue(res.is_grounded)
        self.assertIn("Paris is the capital of France.", res.final_answer)
        self.assertIn("Lyon is the third-largest city in France.", res.final_answer)

        # 2. Contradicted claim is NOT asserted as fact
        self.assertNotIn("Lyon is the largest city in France", res.final_answer)
        self.assertIn("c_1", [c for c in assmts if assmts[c].label == AssessmentLabel.CONTRADICTED])
        self.assertTrue(any("largest" in c for c in res.contradicted_claims))

        # 3. Preserves evidence provenance / citations for verified claims
        self.assertIn("Paris Overview", res.cited_sources)
        self.assertIn("Lyon Overview", res.cited_sources)
        self.assertIn("[Paris Overview]", res.final_answer)
        self.assertIn("[Lyon Overview]", res.final_answer)

        # 4. Correct ordering: Paris (c_0) precedes Lyon (c_2) in the synthesized text
        pos_c0 = res.final_answer.find("Paris is the capital of France.")
        pos_c2 = res.final_answer.find("Lyon is the third-largest city in France.")
        self.assertGreater(pos_c2, pos_c0, "Verified claims must appear in original claim order")


    # 10. Successful repair path: contradicted claim -> grounded repair revision -> synthesized into final answer with citations
    def test_successful_repair_incorporation_in_synthesis(self):
        """Task 3 verification: Verify a successful repair path where the repairer
        receives a contradicted claim, produces a valid revision grounded in supplied
        evidence, and the synthesizer incorporates that revision into the final answer
        with provenance.
        """
        c_orig = ClaimRecord(
            claim_id="c_rep",
            source_sentence_id="s_0",
            original_text="Lyon is the capital of France.",
            normalized_text="Lyon is the capital of France.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.UNRESOLVED,
        )

        passage = EvidencePassage(
            evidence_id="ev_paris_101",
            doc_title="French Capitals Guide",
            source_text="Paris is the official capital and administrative center of France. Lyon is a major commercial hub.",
            retrieval_score=0.95,
        )

        assmt = EvidenceAssessment(
            claim_id="c_rep",
            evidence_ids=["ev_paris_101"],
            supporting_spans={"ev_paris_101": ["Paris is the official capital"]},
            label=AssessmentLabel.CONTRADICTED,
            confidence=0.92,
            rationale="Passage confirms Paris is capital, contradicting Lyon.",
        )

        # Build mock LLM response payload for LLMRepairer
        repair_payload = json.dumps({
            "revised_claim": "Paris is the official capital of France.",
            "repair_reason": "Corrected city to Paris based on French Capitals Guide.",
            "confidence": 0.95,
        })
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = repair_payload
        mock_resp.usage.prompt_tokens = 60
        mock_resp.usage.completion_tokens = 25
        mock_client.chat.completions.create.return_value = mock_resp

        repairer = LLMRepairer(client=mock_client, model_name="local_repairer")
        repair_res, cost = repairer.repair(c_orig, assmt, [passage])

        # Verify repairer produced a valid grounded revision
        self.assertEqual(repair_res.claim_id, "c_rep")
        self.assertEqual(repair_res.proposed_revision, "Paris is the official capital of France.")
        self.assertIn("ev_paris_101", repair_res.supporting_evidence_ids)
        self.assertGreaterEqual(repair_res.confidence, 0.9)

        # A repair is only presented as grounded once its revised text was re-verified.
        repair_res.reverification = {"label": "SUPPORTED", "confidence": 0.9, "verified": True}

        # Build trace step representing the executed repair action
        trace = ControllerTrace(
            run_id="test_repair_run",
            policy_name="TestPolicy",
            policy_config={},
            initial_budget={},
            final_budget_remaining={},
            total_cost={},
            total_steps=1,
            stopping_reason=StoppingReason.ALL_CLAIMS_RESOLVED,
            claims_summary={"UNRESOLVED": 1},
            steps=[
                TraceStep(
                    step_number=1,
                    claim_id="c_rep",
                    claim_text=c_orig.original_text,
                    triage_score=0.8,
                    previous_state=ClaimState.SELECTED.value,
                    new_state=ClaimState.UNRESOLVED.value,
                    action=ControllerAction.REPAIR,
                    selection_rationale="Contradicted claim eligible for repair",
                    action_outcome=f"Claim repaired: '{repair_res.proposed_revision}'",
                    assessment_label=AssessmentLabel.CONTRADICTED.value,
                    repair_data=repair_res.to_dict(),
                )
            ],
        )

        # Synthesize with MockAnswerSynthesizer
        synth = MockAnswerSynthesizer()
        synth_res = synth.synthesize(
            question="What is the capital of France?",
            initial_draft_answer="Lyon is the capital of France.",
            claims=[c_orig],
            evidence_store={"c_rep": [passage]},
            assessments={"c_rep": assmt},
            trace=trace,
        )

        # Verify final answer incorporates the repaired revision with source provenance
        # Note: repaired claims are qualified/corrected but not evidence-VERIFIED,
        # so is_grounded remains False while has_unresolved_limitations is True.
        self.assertFalse(synth_res.is_grounded)
        self.assertTrue(synth_res.has_unresolved_limitations)
        self.assertIn("Paris is the official capital of France.", synth_res.final_answer)
        self.assertNotIn("Lyon is the capital of France.", synth_res.final_answer)

        # Same trace, but re-verification did not pass: the revision must not be asserted.
        repair_res.reverification = {"label": "INSUFFICIENT", "confidence": 0.4, "verified": False}
        trace.steps[0].repair_data = repair_res.to_dict()
        synth_res2 = synth.synthesize(
            question="What is the capital of France?",
            initial_draft_answer="Lyon is the capital of France.",
            claims=[c_orig],
            evidence_store={"c_rep": [passage]},
            assessments={"c_rep": assmt},
            trace=trace,
        )
        self.assertNotIn("Paris is the official capital of France.", synth_res2.final_answer)
        self.assertFalse(synth_res2.is_grounded)
        self.assertIn("French Capitals Guide", synth_res.cited_sources)
        self.assertIn("[French Capitals Guide]", synth_res.final_answer)


if __name__ == "__main__":
    unittest.main()


