"""Comprehensive regression tests for verification controller, assessor, policy, and synthesis.

Covers the 13 required regression criteria:
1. Joint multi-passage entailment prompt and span attribution
2. Irrelevant passages yield INSUFFICIENT
3. Contradictory passages yield CONTRADICTED/CONFLICTED
4. Missing evidence yields INSUFFICIENT
5. Evidence provenance and citation validity (no init_query or unverified claim leakage)
6. Synthesis refuses unsupported claims (safe abstention, is_grounded=False)
7. Safe abstention does NOT echo draft answer when unverified
8. Truncated synthesis output retry logic
9. ALL_CLAIMS_RESOLVED only when no UNRESOLVED claims remain
10. Duplicate retrieval stall detection suppresses unproductive RETRIEVE
11. Budget exhaustion yields BUDGET_EXHAUSTED
12. MockRepairer consumes 0 model calls
13. Token and cost totals are strictly additive across pipeline stages
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus
from src.controller.budget import BudgetTracker
from src.controller.controller import VerificationController
from src.controller.interfaces import MockRepairer, MockRetriever
from src.controller.models import (
    ControllerAction,
    ControllerTrace,
    PerClaimBudget,
    StoppingReason,
    TraceStep,
    VerificationBudget,
)
from src.controller.policy import ControllerPolicyConfig, SequentialDecisionPolicy, StaticAllocationPolicy, StaticCTSRankedPolicy
from src.evidence.assessor import MockDeterministicAssessor
from src.evidence.llama_assessor import ASSESSMENT_PROMPT, LlamaAssessor
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost
from src.pipeline import PipelineResult, RAGVerificationPipeline
from src.synthesis.models import SynthesisResult
from src.synthesis.synthesizer import LLMAnswerSynthesizer, MockAnswerSynthesizer


def _create_mock_client(content: str, prompt_tokens: int = 100, completion_tokens: int = 30, finish_reason: str = "stop"):
    mock_client = MagicMock()
    mock_model = MagicMock()
    mock_model.id = "test_model"
    mock_client.models.list.return_value = MagicMock(data=[mock_model])
    mock_msg = MagicMock()
    mock_msg.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_choice.finish_reason = finish_reason
    mock_response = MagicMock(
        choices=[mock_choice],
        usage=MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens),
    )
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


class TestJointEvidenceAndAssessorRegressions(unittest.TestCase):
    """Regression tests for joint multi-passage reasoning and evidence assessment."""

    def test_joint_multi_passage_reasoning_prompt_and_parsing(self):
        """1. Joint multi-passage entailment prompt and span attribution."""
        self.assertIn("reason across multiple passages jointly", ASSESSMENT_PROMPT)
        self.assertIn("passage_id", ASSESSMENT_PROMPT)

        resp_content = json.dumps({
            "label": "SUPPORTED",
            "supporting_spans": [
                {"passage_id": "p1", "text": "General Manager of Disney's Aulani Resort & Spa"},
                {"passage_id": "p2", "text": "located in Kapolei on Oahu, Hawaii"},
            ],
            "contradicting_spans": [],
            "reasoning": "Passage p1 confirms Djuan Rivers managed Aulani, and passage p2 confirms Aulani is in Hawaii.",
        })
        client = _create_mock_client(resp_content, prompt_tokens=120, completion_tokens=40)
        assessor = LlamaAssessor(client=client, model_name="test_model")

        passages = [
            EvidencePassage(
                evidence_id="p1",
                doc_title="Djuan Rivers Bio",
                source_text="Djuan Rivers was a Vice President and General Manager of Disney's Aulani Resort & Spa.",
            ),
            EvidencePassage(
                evidence_id="p2",
                doc_title="Aulani Resort",
                source_text="Aulani, a Disney Resort & Spa, is a beachside hotel and resort located in Kapolei on Oahu, Hawaii.",
            ),
        ]
        claim = ClaimRecord(
            claim_id="c_hawaii",
            source_sentence_id="s1",
            original_text="The Disney Resort that Djuan Rivers managed is in Hawaii.",
            normalized_text="The Disney Resort that Djuan Rivers managed is in Hawaii.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
        )
        assessment = assessor.assess(claim, passages)

        self.assertEqual(assessment.label, AssessmentLabel.SUPPORTED)
        self.assertIn("p1", assessment.supporting_spans)
        self.assertIn("p2", assessment.supporting_spans)
        self.assertEqual(len(assessment.supporting_spans["p1"]), 1)
        self.assertEqual(len(assessment.supporting_spans["p2"]), 1)
        self.assertGreaterEqual(assessment.cost.model_calls, 1)

    def test_irrelevant_passages_yield_insufficient(self):
        """2. Irrelevant passages yield INSUFFICIENT."""
        resp_content = json.dumps({
            "label": "INSUFFICIENT",
            "supporting_spans": [],
            "contradicting_spans": [],
            "reasoning": "Mentions Grand Floridian in Florida, not Djuan Rivers or Hawaii.",
        })
        client = _create_mock_client(resp_content, prompt_tokens=80, completion_tokens=25)
        assessor = LlamaAssessor(client=client, model_name="test_model")

        passages = [
            EvidencePassage(
                evidence_id="p_other",
                doc_title="Grand Floridian",
                source_text="Disney's Grand Floridian Resort & Spa is located in Florida near Orlando.",
            )
        ]
        claim = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Djuan Rivers was General Manager of a Hawaii Disney resort.",
            normalized_text="Djuan Rivers was General Manager of a Hawaii Disney resort.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
        )
        assessment = assessor.assess(claim, passages)
        self.assertEqual(assessment.label, AssessmentLabel.INSUFFICIENT)
        self.assertEqual(len(assessment.supporting_spans), 0)

    def test_contradictory_passages_yield_conflicted(self):
        """3. Contradictory passages yield CONTRADICTED or CONFLICTED."""
        resp_content = json.dumps({
            "label": "CONTRADICTED",
            "supporting_spans": [],
            "contradicting_spans": [
                {"passage_id": "p_contra", "text": "spent his entire hospitality career working exclusively in Europe"}
            ],
            "reasoning": "Passage directly contradicts managing a resort in Hawaii.",
        })
        client = _create_mock_client(resp_content, prompt_tokens=90, completion_tokens=30)
        assessor = LlamaAssessor(client=client, model_name="test_model")

        passages = [
            EvidencePassage(
                evidence_id="p_contra",
                doc_title="Djuan Rivers Career",
                source_text="Djuan Rivers spent his entire hospitality career working exclusively in Europe.",
            )
        ]
        claim = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Djuan Rivers managed a Disney resort in Hawaii.",
            normalized_text="Djuan Rivers managed a Disney resort in Hawaii.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
        )
        assessment = assessor.assess(claim, passages)
        self.assertIn(assessment.label, (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED))

    def test_missing_evidence_yields_insufficient(self):
        """4. Missing evidence (empty passages list) yields INSUFFICIENT immediately."""
        client = _create_mock_client("{}")
        assessor = LlamaAssessor(client=client, model_name="test_model")
        claim = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Some ungrounded claim.",
            normalized_text="Some ungrounded claim.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
        )
        assessment = assessor.assess(claim, [])
        self.assertEqual(assessment.label, AssessmentLabel.INSUFFICIENT)
        self.assertEqual(assessment.cost.model_calls, 0)


class TestGroundedSynthesisRegressions(unittest.TestCase):
    """Regression tests for answer synthesis and grounding enforcement."""

    def setUp(self):
        self.question = "What State has a Disney Resort & Spa that Djuan Rivers was a General Manager at?"
        self.draft_answer = "The state is Hawaii."

    def test_evidence_provenance_and_citation_validity(self):
        """5. Synthesis cited_sources strictly includes verified-claim passages, excluding init_query."""
        client = _create_mock_client("Based on verified evidence, the state is Hawaii [Aulani Resort Guide].")
        synthesizer = LLMAnswerSynthesizer(client=client, model_name="test_model")

        verified_claim = ClaimRecord(
            claim_id="c_verified",
            source_sentence_id="s1",
            original_text="Aulani Disney Resort is in Hawaii.",
            normalized_text="Aulani Disney Resort is in Hawaii.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
        )
        unverified_claim = ClaimRecord(
            claim_id="c_unresolved",
            source_sentence_id="s2",
            original_text="Djuan Rivers was born in California.",
            normalized_text="Djuan Rivers was born in California.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.UNRESOLVED,
        )
        evidence_store = {
            "init_query": [
                EvidencePassage(evidence_id="init_1", doc_title="Unrelated Global Doc", source_text="Random text.")
            ],
            "c_verified": [
                EvidencePassage(evidence_id="ev_aulani", doc_title="Aulani Resort Guide", source_text="Aulani is in Hawaii.")
            ],
            "c_unresolved": [
                EvidencePassage(evidence_id="ev_california", doc_title="California History", source_text="California info.")
            ],
        }

        res = synthesizer.synthesize(
            question=self.question,
            initial_draft_answer=self.draft_answer,
            claims=[verified_claim, unverified_claim],
            evidence_store=evidence_store,
        )
        # Must cite Aulani Resort Guide
        self.assertIn("Aulani Resort Guide", res.cited_sources)
        # Must NOT cite init_query dummy doc or unverified claim docs
        self.assertNotIn("Unrelated Global Doc", res.cited_sources)
        self.assertNotIn("California History", res.cited_sources)

    def test_synthesis_refuses_unsupported_claims_safe_abstention(self):
        """6. Safe abstention when all claims are UNRESOLVED (is_grounded=False, no LLM call)."""
        client = _create_mock_client("Should not be called")
        synthesizer = LLMAnswerSynthesizer(client=client, model_name="test_model")

        unresolved_claim = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="The state is Hawaii.",
            normalized_text="The state is Hawaii.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.UNRESOLVED,
        )
        evidence_store = {"c1": [EvidencePassage(evidence_id="ev1", doc_title="Doc", source_text="Text")]}

        res = synthesizer.synthesize(
            question=self.question,
            initial_draft_answer=self.draft_answer,
            claims=[unresolved_claim],
            evidence_store=evidence_store,
        )
        # Safe abstention path must NOT invoke remote LLM
        client.chat.completions.create.assert_not_called()
        self.assertFalse(res.is_grounded)
        self.assertTrue(res.has_unresolved_limitations)
        self.assertIn("insufficient", res.final_answer.lower())
        self.assertEqual(res.synthesis_method, "safe_abstention")

    def test_no_verified_claims_does_not_echo_draft(self):
        """7. Even if draft has confident hallucination, zero verified claims does NOT echo draft."""
        synthesizer = LLMAnswerSynthesizer()
        res = synthesizer.synthesize(
            question=self.question,
            initial_draft_answer="Djuan Rivers was General Manager at Aulani in Hawaii.",
            claims=[],  # Zero verified claims
            evidence_store={},
        )
        self.assertFalse(res.is_grounded)
        self.assertNotIn("Djuan Rivers was General Manager at Aulani", res.final_answer)
        self.assertIn("insufficient", res.final_answer.lower())

    def test_truncated_synthesis_output_retry(self):
        """8. Truncated synthesis output (finish_reason == 'length') triggers retry."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "test_model"
        mock_client.models.list.return_value = MagicMock(data=[mock_model])

        msg1 = MagicMock(content="The state is")
        choice1 = MagicMock(message=msg1, finish_reason="length")
        resp1 = MagicMock(choices=[choice1], usage=MagicMock(prompt_tokens=50, completion_tokens=5, total_tokens=55))

        msg2 = MagicMock(content="The state is Hawaii based on the verified evidence.")
        choice2 = MagicMock(message=msg2, finish_reason="stop")
        resp2 = MagicMock(choices=[choice2], usage=MagicMock(prompt_tokens=50, completion_tokens=15, total_tokens=65))

        mock_client.chat.completions.create.side_effect = [resp1, resp2]
        synthesizer = LLMAnswerSynthesizer(client=mock_client, model_name="test_model")

        verified_claim = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Hawaii has the Aulani resort.",
            normalized_text="Hawaii has the Aulani resort.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
        )
        evidence_store = {
            "c1": [EvidencePassage(evidence_id="p1", doc_title="Doc1", source_text="Hawaii has Aulani.")]
        }

        res = synthesizer.synthesize(
            question=self.question,
            initial_draft_answer=self.draft_answer,
            claims=[verified_claim],
            evidence_store=evidence_store,
        )
        # Verify retry was attempted (create called twice)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        self.assertEqual(res.cost.model_calls, 2)
        self.assertIn("Hawaii", res.final_answer)


class TestControllerAndPolicyRegressions(unittest.TestCase):
    """Regression tests for controller stopping condition, stall prevention, and cost accounting."""

    def test_stopping_reason_all_claims_resolved_vs_unresolved(self):
        """9. ALL_CLAIMS_RESOLVED only when no UNRESOLVED claims remain; otherwise NO_ELIGIBLE_ACTIONS."""
        budget = VerificationBudget(max_steps=10)

        # Case A: One claim UNRESOLVED
        c_unresolved = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Unresolved text",
            normalized_text="Unresolved text",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.UNRESOLVED,
        )
        ctrl = VerificationController(
            policy=SequentialDecisionPolicy(),
            assessor=MockDeterministicAssessor(),
            retriever=MockRetriever(),
            global_budget=budget,
        )
        trace_a = ctrl.run([c_unresolved], {})
        # Because claim is UNRESOLVED, stopping reason MUST NOT be ALL_CLAIMS_RESOLVED
        self.assertNotEqual(trace_a.stopping_reason, StoppingReason.ALL_CLAIMS_RESOLVED)
        self.assertEqual(trace_a.stopping_reason, StoppingReason.NO_ELIGIBLE_ACTIONS)

        # Case B: Claim is VERIFIED -> ALL_CLAIMS_RESOLVED
        c_verified = ClaimRecord(
            claim_id="c2",
            source_sentence_id="s1",
            original_text="Verified text",
            normalized_text="Verified text",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
        )
        ctrl_b = VerificationController(
            policy=SequentialDecisionPolicy(),
            assessor=MockDeterministicAssessor(),
            retriever=MockRetriever(),
            global_budget=budget,
        )
        trace_b = ctrl_b.run([c_verified], {})
        self.assertEqual(trace_b.stopping_reason, StoppingReason.ALL_CLAIMS_RESOLVED)

    def test_duplicate_retrieval_stall_suppresses_retrieve(self):
        """10. Duplicate retrieval yielding 0 unique passages suppresses subsequent RETRIEVE."""
        budget = VerificationBudget(max_steps=10)
        per_claim = PerClaimBudget(max_retrievals=5)
        tracker = BudgetTracker(global_budget=budget, per_claim_budget=per_claim)

        c = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Some text",
            normalized_text="Some text",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.SELECTED,
        )
        passage = EvidencePassage(evidence_id="p1", doc_title="Title", source_text="Text")
        assmt = EvidenceAssessment(
            claim_id="c1",
            evidence_ids=["p1"],
            supporting_spans={},
            label=AssessmentLabel.INSUFFICIENT,
            confidence=0.5,
            rationale="Inconclusive evidence passages.",
        )
        evidence_store = {"c1": [passage]}
        assessments = {"c1": assmt}

        # Record a prior retrieval that yielded 0 unique passages
        tracker.record_action(ControllerAction.RETRIEVE, "c1")
        tracker.record_retrieval_passages("c1", unique_count=0, duplicate_count=2)

        # SequentialDecisionPolicy should suppress RETRIEVE
        policy_seq = SequentialDecisionPolicy()
        action_seq, claim_seq, _ = policy_seq.select_next_action(
            claims=[c], evidence_store=evidence_store, assessments=assessments, repairs={}, budget_tracker=tracker
        )
        self.assertNotEqual(action_seq, ControllerAction.RETRIEVE, "SequentialDecisionPolicy should not schedule RETRIEVE after 0 unique passages")

        # StaticAllocationPolicy should suppress RETRIEVE
        policy_fifo = StaticAllocationPolicy()
        action_fifo, _, _ = policy_fifo.select_next_action(
            claims=[c], evidence_store=evidence_store, assessments=assessments, repairs={}, budget_tracker=tracker
        )
        self.assertNotEqual(action_fifo, ControllerAction.RETRIEVE, "StaticAllocationPolicy should not schedule RETRIEVE after 0 unique passages")

        # StaticCTSRankedPolicy should suppress RETRIEVE
        policy_cts = StaticCTSRankedPolicy()
        action_cts, _, _ = policy_cts.select_next_action(
            claims=[c], evidence_store=evidence_store, assessments=assessments, repairs={}, budget_tracker=tracker
        )
        self.assertNotEqual(action_cts, ControllerAction.RETRIEVE, "StaticCTSRankedPolicy should not schedule RETRIEVE after 0 unique passages")

    def test_budget_exhaustion_stopping_reason(self):
        """11. Budget exhaustion yields BUDGET_EXHAUSTED."""
        budget = VerificationBudget(max_model_calls=0)
        c = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Claim text",
            normalized_text="Claim text",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.PENDING,
        )
        ctrl = VerificationController(
            policy=SequentialDecisionPolicy(),
            assessor=MockDeterministicAssessor(),
            retriever=MockRetriever(),
            global_budget=budget,
        )
        trace = ctrl.run([c], {})
        self.assertEqual(trace.stopping_reason, StoppingReason.BUDGET_EXHAUSTED)

    def test_mock_repairer_zero_model_calls(self):
        """12. MockRepairer consumes 0 model calls."""
        repairer = MockRepairer()
        c = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Contradicted text",
            normalized_text="Contradicted text",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.PENDING,
        )
        assmt = EvidenceAssessment(
            claim_id="c1",
            evidence_ids=["p1"],
            supporting_spans={},
            label=AssessmentLabel.CONTRADICTED,
            confidence=0.9,
            rationale="Passage contradicts claim",
        )
        repair_res, cost = repairer.repair(c, assmt, [])
        self.assertEqual(cost.model_calls, 0)
        self.assertIn("[Qualified]:", repair_res.proposed_revision)

    def test_token_totals_additive(self):
        """13. Token and cost totals are strictly additive across initial retrieval, verification, and synthesis."""
        pipeline = RAGVerificationPipeline(dry_run=True)
        res = pipeline.run("What is the capital of France?", policy_name="StaticFIFO")

        tot = res.total_cost
        init = res.initial_cost
        ctrl = res.verification_cost
        synth = res.synthesis_cost

        # Model calls must equal init + ctrl + synth
        expected_calls = init.get("model_calls", 0) + ctrl.get("model_calls", 0) + synth.get("model_calls", 0)
        self.assertEqual(tot["model_calls"], expected_calls)

        # Total tokens must equal init + ctrl + synth
        expected_tokens = init.get("total_tokens", 0) + ctrl.get("total_tokens", 0) + synth.get("total_tokens", 0)
        self.assertEqual(tot["total_tokens"], expected_tokens)




class TestQueryAwareStallAndSynthesisFixes(unittest.TestCase):
    """Regression tests for the 2026-10-26 targeted fixes:
    - Query-aware retrieval stall detection
    - Sentence-boundary trimming on truncated synthesis output
    - Irrelevant non-supporting passages excluded from citations
    """

    # --------------------------------------------------------------------------
    # 14. Query-aware stall detection: changed query un-stalls retrieval
    # --------------------------------------------------------------------------

    def test_query_change_unstalls_retrieval_in_budget_tracker(self):
        """14a. is_retrieval_stalled returns False when query changes after 0-unique retrieval."""
        tracker = BudgetTracker(
            global_budget=VerificationBudget(max_steps=20),
            per_claim_budget=PerClaimBudget(max_retrievals=5),
        )
        cid = "c_stall"
        original_query = "Djuan Rivers Disney resort state"
        new_query = "Aulani beachside hotel Hawaii location"

        # Simulate a first retrieval with original query that yielded 0 unique passages
        tracker.record_action(ControllerAction.RETRIEVE, cid)
        tracker.record_retrieval_passages(cid, unique_count=0, duplicate_count=3, query=original_query)

        # Same query: stalled
        self.assertTrue(
            tracker.is_retrieval_stalled(cid, current_query=original_query),
            "Same query after 0-unique retrieval should be reported as stalled",
        )
        # Different query: not stalled
        self.assertFalse(
            tracker.is_retrieval_stalled(cid, current_query=new_query),
            "Changed query after 0-unique retrieval should NOT be reported as stalled",
        )

    def test_no_prior_retrieval_is_not_stalled(self):
        """14b. is_retrieval_stalled returns False when no retrieval has been recorded."""
        tracker = BudgetTracker()
        self.assertFalse(tracker.is_retrieval_stalled("c_new", current_query="any query"))

    def test_positive_unique_count_is_not_stalled(self):
        """14c. is_retrieval_stalled returns False when prior retrieval yielded >0 unique passages."""
        tracker = BudgetTracker()
        tracker.record_retrieval_passages("c1", unique_count=3, duplicate_count=0, query="some query")
        self.assertFalse(tracker.is_retrieval_stalled("c1", current_query="some query"))

    def test_mab_policy_permits_retrieve_on_changed_query(self):
        """14d. MABLinUCBPolicy permits RETRIEVE after 0-unique retrieval if query text changed.

        The claim's normalized_text is used as the query proxy.  We simulate the
        scenario by recording a stall against a *different* effective query so that
        the current claim's text differs, making is_retrieval_stalled return False.
        """
        from src.controller.mab_policy import ContextualMABPolicy, MABPolicyConfig

        tracker = BudgetTracker(
            global_budget=VerificationBudget(max_steps=20),
            per_claim_budget=PerClaimBudget(max_retrievals=5),
        )
        cid = "c_mab"
        # Record a stall against a different query (simulating a prior different query attempt)
        tracker.record_action(ControllerAction.RETRIEVE, cid)
        tracker.record_retrieval_passages(cid, unique_count=0, duplicate_count=1, query="old query text")

        # The claim's normalized_text acts as the current query in the MAB policy
        c = ClaimRecord(
            claim_id=cid,
            source_sentence_id="s1",
            original_text="new query text",
            normalized_text="new query text",  # different from "old query text"
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.SELECTED,
        )
        assmt = EvidenceAssessment(
            claim_id=cid,
            evidence_ids=["p1"],
            supporting_spans={},
            label=AssessmentLabel.INSUFFICIENT,
            confidence=0.5,
            rationale="Insufficient evidence.",
        )
        evidence_store = {"c_mab": [EvidencePassage(evidence_id="p1", doc_title="Doc", source_text="Text")]}
        assessments = {cid: assmt}

        policy = ContextualMABPolicy(config=MABPolicyConfig(enable_repair=False))
        action, _, _ = policy.select_next_action(
            claims=[c],
            evidence_store=evidence_store,
            assessments=assessments,
            repairs={},
            budget_tracker=tracker,
        )
        # Policy should allow RETRIEVE because the query changed
        self.assertEqual(
            action, ControllerAction.RETRIEVE,
            "MABLinUCBPolicy should permit RETRIEVE when claim query differs from prior stalled query",
        )

    # --------------------------------------------------------------------------
    # 15. Sentence-boundary trimming on truncated synthesis
    # --------------------------------------------------------------------------

    def test_trim_to_last_sentence_basic(self):
        """15a. _trim_to_last_sentence trims to last sentence-ending punctuation."""
        from src.synthesis.synthesizer import _trim_to_last_sentence

        text = "The state is Hawaii. The resort is located by a palm-tree lined covered"
        result = _trim_to_last_sentence(text)
        self.assertEqual(result, "The state is Hawaii.")

    def test_trim_to_last_sentence_question_mark(self):
        """15b. _trim_to_last_sentence handles '?' as boundary."""
        from src.synthesis.synthesizer import _trim_to_last_sentence

        text = "Did the evidence confirm Hawaii? The next relevant fact was"
        result = _trim_to_last_sentence(text)
        self.assertEqual(result, "Did the evidence confirm Hawaii?")

    def test_trim_to_last_sentence_no_boundary_returns_original(self):
        """15c. _trim_to_last_sentence returns original when no sentence boundary exists."""
        from src.synthesis.synthesizer import _trim_to_last_sentence

        text = "No sentence boundary here"
        result = _trim_to_last_sentence(text)
        self.assertEqual(result, text)

    def test_synthesis_truncation_salvages_complete_sentence(self):
        """15d. When LLM returns finish_reason='length', synthesizer trims to last complete sentence.

        The salvaged sentence should be accepted rather than retrying,
        so create() is called only once.
        """
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "test_model"
        mock_client.models.list.return_value = MagicMock(data=[mock_model])

        # Response ends mid-sentence after a complete sentence
        truncated_content = "The state is Hawaii. The resort is located by a palm-tree lined covered"
        choice = MagicMock(message=MagicMock(content=truncated_content), finish_reason="length")
        resp = MagicMock(
            choices=[choice],
            usage=MagicMock(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        )
        mock_client.chat.completions.create.return_value = resp

        synthesizer = LLMAnswerSynthesizer(client=mock_client, model_name="test_model")
        verified_claim = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Aulani is in Hawaii.",
            normalized_text="Aulani is in Hawaii.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
        )
        evidence_store = {
            "c1": [EvidencePassage(evidence_id="p1", doc_title="Aulani Doc", source_text="Aulani is in Hawaii.")]
        }
        res = synthesizer.synthesize(
            question="What state is Aulani in?",
            initial_draft_answer="Hawaii",
            claims=[verified_claim],
            evidence_store=evidence_store,
        )
        # Sentence-boundary trimming should have salvaged a complete answer; only 1 LLM call needed
        self.assertEqual(mock_client.chat.completions.create.call_count, 1,
                         "Should NOT retry when truncated content has a salvageable complete sentence")
        # Answer must end at a sentence boundary (no dangling fragment)
        self.assertTrue(
            res.final_answer.endswith(".") or res.final_answer.endswith("!") or res.final_answer.endswith("?"),
            f"Final answer should end at sentence boundary, got: '{res.final_answer}'",
        )
        # The complete sentence from before the mid-sentence truncation must be present
        self.assertIn("Hawaii", res.final_answer)

    # --------------------------------------------------------------------------
    # 16. Irrelevant passages excluded from synthesis citations and context
    # --------------------------------------------------------------------------

    def test_irrelevant_passages_excluded_from_citations(self):
        """16a. Passages in evidence_store for NON-verified claims must NOT appear in cited_sources."""
        client = _create_mock_client("Based on verified evidence, the state is Hawaii [Aulani Resort].")
        synthesizer = LLMAnswerSynthesizer(client=client, model_name="test_model")

        verified_claim = ClaimRecord(
            claim_id="c_verified",
            source_sentence_id="s1",
            original_text="Aulani is in Hawaii.",
            normalized_text="Aulani is in Hawaii.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
        )
        unresolved_claim = ClaimRecord(
            claim_id="c_unresolved",
            source_sentence_id="s2",
            original_text="Some unrelated claim.",
            normalized_text="Some unrelated claim.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.UNRESOLVED,
        )
        evidence_store = {
            "c_verified": [
                EvidencePassage(evidence_id="p_good", doc_title="Aulani Resort", source_text="Aulani in Hawaii.")
            ],
            "c_unresolved": [
                EvidencePassage(evidence_id="p_irrelevant_1", doc_title="Walt Disney World Dolphin", source_text="Resort in Florida."),
                EvidencePassage(evidence_id="p_irrelevant_2", doc_title="Saratoga Springs Resort", source_text="Another resort."),
            ],
        }
        res = synthesizer.synthesize(
            question="What state is Aulani in?",
            initial_draft_answer="Hawaii",
            claims=[verified_claim, unresolved_claim],
            evidence_store=evidence_store,
        )
        self.assertIn("Aulani Resort", res.cited_sources)
        self.assertNotIn("Walt Disney World Dolphin", res.cited_sources,
                         "Irrelevant resort passage from unresolved claim must not appear in citations")
        self.assertNotIn("Saratoga Springs Resort", res.cited_sources,
                         "Irrelevant resort passage from unresolved claim must not appear in citations")

    def test_supporting_span_restricts_citations_further(self):
        """16b. When supporting_spans are present, only passages with matching IDs are cited."""
        client = _create_mock_client("Based on evidence, the answer is Hawaii [Aulani Resort].")
        synthesizer = LLMAnswerSynthesizer(client=client, model_name="test_model")

        # Claim has supporting span only from p1, not p2
        verified_claim = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Aulani is in Hawaii.",
            normalized_text="Aulani is in Hawaii.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
            evidence_provenance={"supporting_spans": {"p1": ["located in Hawaii"]}},
        )
        evidence_store = {
            "c1": [
                EvidencePassage(evidence_id="p1", doc_title="Aulani Resort", source_text="Aulani located in Hawaii."),
                EvidencePassage(evidence_id="p2", doc_title="Unrelated Extra Doc", source_text="Some other fact."),
            ]
        }
        res = synthesizer.synthesize(
            question="What state is Aulani in?",
            initial_draft_answer="Hawaii",
            claims=[verified_claim],
            evidence_store=evidence_store,
        )
        self.assertIn("Aulani Resort", res.cited_sources)
        self.assertNotIn("Unrelated Extra Doc", res.cited_sources,
                         "Passage without a supporting span entry must not be cited")


if __name__ == "__main__":
    unittest.main()
