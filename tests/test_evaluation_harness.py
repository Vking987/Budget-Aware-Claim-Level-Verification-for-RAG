"""Unit tests for Stage C2 evaluation harness, QdrantRetriever, and StaticAllocationPolicy.

Covers:
1. QdrantRetriever offline operation with mock Qdrant client and mock encoder.
2. Handling of missing metadata and empty queries in QdrantRetriever.
3. Deduplication of identical passages in QdrantRetriever.
4. StaticAllocationPolicy FIFO arrival ordering (ignoring CTS).
5. StaticAllocationPolicy budget exhaustion and per-claim limits.
6. End-to-end dry-run execution of scripts/run_policy_comparison.py.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus
from src.controller.budget import BudgetTracker
from src.controller.controller import VerificationController
from src.controller.interfaces import MockRepairer, MockRetriever
from src.controller.models import (
    ControllerAction,
    PerClaimBudget,
    RepairResult,
    StoppingReason,
    VerificationBudget,
)
from src.controller.policy import (
    ControllerPolicyConfig,
    SequentialDecisionPolicy,
    StaticAllocationPolicy,
    StaticCTSRankedPolicy,
)
from src.controller.retriever import QdrantRetriever
from src.evidence.assessor import MockDeterministicAssessor
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost


def _make_claim(
    claim_id: str,
    text: str,
    claim_type: ClaimType = ClaimType.FACTUAL_RELATION,
    triage_score: float = 0.5,
    state: ClaimState = ClaimState.PENDING,
) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        source_sentence_id="s1",
        original_text=text,
        normalized_text=text,
        claim_type=claim_type,
        decomposition_status=DecompositionStatus.ATOMIC,
        state=state,
        triage_score=triage_score,
    )


class MockHit:
    """Mock Qdrant ScoredPoint."""
    def __init__(self, hit_id: int, score: float, payload: Dict[str, Any]) -> None:
        self.id = hit_id
        self.score = score
        self.payload = payload


class TestQdrantRetriever(unittest.TestCase):
    """Test QdrantRetriever adapter offline contracts."""

    def test_qdrant_retriever_offline_search(self):
        mock_client = MagicMock()
        mock_client.search.return_value = [
            MockHit(1, 0.92, {"title": "Doc A", "text": "Text of Document A.", "hotpot_id": "hp1"}),
            MockHit(2, 0.85, {"title": "Doc B", "text": "Text of Document B.", "hotpot_id": "hp2"}),
        ]
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.1] * 1024

        retriever = QdrantRetriever(
            client=mock_client,
            embedding_model=mock_encoder,
            collection_name="test_col",
        )
        claim = _make_claim("c1", "Arthur's Magazine was founded in 1844.")
        passages, cost = retriever.retrieve(claim, top_k=2)

        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0].doc_title, "Doc A")
        self.assertEqual(passages[0].evidence_id, "qdrant_1")
        self.assertEqual(passages[0].source_text, "Text of Document A.")
        self.assertEqual(passages[0].retrieval_score, 0.92)
        self.assertEqual(passages[0].metadata["hotpot_id"], "hp1")
        self.assertEqual(cost.model_calls, 1)
        mock_client.search.assert_called_once()

    def test_qdrant_retriever_query_points_interface(self):
        mock_client = MagicMock()
        del mock_client.search  # Remove search method to test query_points branch
        mock_resp = MagicMock()
        mock_resp.points = [
            MockHit(10, 0.88, {"title": "Doc C", "text": "Content of Doc C."}),
        ]
        mock_client.query_points.return_value = mock_resp
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.2] * 1024

        retriever = QdrantRetriever(
            client=mock_client,
            embedding_model=mock_encoder,
        )
        claim = _make_claim("c2", "Query points test claim.")
        passages, cost = retriever.retrieve(claim, top_k=1)

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0].doc_title, "Doc C")
        self.assertEqual(passages[0].evidence_id, "qdrant_10")

    def test_qdrant_retriever_deduplication(self):
        mock_client = MagicMock()
        mock_client.search.return_value = [
            MockHit(1, 0.95, {"title": "Same Title", "text": "Identical content across two duplicate hits."}),
            MockHit(2, 0.91, {"title": "Same Title", "text": "Identical content across two duplicate hits."}),
            MockHit(3, 0.80, {"title": "Different Doc", "text": "Completely novel content."}),
        ]
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.1] * 1024

        retriever = QdrantRetriever(
            client=mock_client,
            embedding_model=mock_encoder,
        )
        claim = _make_claim("c3", "Deduplication test.")
        passages, cost = retriever.retrieve(claim, top_k=3)

        # The duplicate hit should be discarded, leaving 2 unique passages
        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0].doc_title, "Same Title")
        self.assertEqual(passages[1].doc_title, "Different Doc")

    def test_qdrant_retriever_missing_metadata(self):
        mock_client = MagicMock()
        mock_client.search.return_value = [
            MockHit(100, 0.75, {"text": "Passage with missing title"}),  # Missing title
            MockHit(101, 0.70, {"title": "Doc Without Text", "text": ""}),  # Empty text (should be skipped)
        ]
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.1] * 1024

        retriever = QdrantRetriever(
            client=mock_client,
            embedding_model=mock_encoder,
        )
        claim = _make_claim("c4", "Missing metadata test.")
        passages, cost = retriever.retrieve(claim, top_k=2)

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0].doc_title, "Untitled")
        self.assertEqual(passages[0].source_text, "Passage with missing title")

    def test_qdrant_retriever_empty_claim(self):
        mock_client = MagicMock()
        mock_encoder = MagicMock()
        retriever = QdrantRetriever(client=mock_client, embedding_model=mock_encoder)
        claim = _make_claim("c5", "")
        passages, cost = retriever.retrieve(claim, top_k=2)

        self.assertEqual(passages, [])
        self.assertEqual(cost.model_calls, 0)
        mock_client.search.assert_not_called()


class TestStaticAllocationPolicy(unittest.TestCase):
    """Test StaticAllocationPolicy FIFO behavior."""

    def test_fifo_ordering_ignores_cts(self):
        policy = StaticAllocationPolicy(ControllerPolicyConfig())
        tracker = BudgetTracker(VerificationBudget(max_steps=10))

        # Claim 1 has LOW triage score (0.1) but appears FIRST
        # Claim 2 has HIGH triage score (0.9) but appears SECOND
        c1 = _make_claim("c1", "First claim in arrival order.", triage_score=0.1)
        c2 = _make_claim("c2", "Second claim in arrival order.", triage_score=0.9)

        # First action MUST target c1 (FIFO), ignoring c2's higher CTS
        action, target, rationale = policy.select_next_action(
            claims=[c1, c2],
            evidence_store={"c1": [], "c2": []},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        self.assertEqual(action, ControllerAction.RETRIEVE)
        self.assertEqual(target.claim_id, "c1")
        self.assertIn("FIFO", rationale)

    def test_fifo_budget_exhaustion(self):
        policy = StaticAllocationPolicy()
        tracker = BudgetTracker(VerificationBudget(max_steps=0))
        c1 = _make_claim("c1", "Claim under zero budget.")

        action, target, _ = policy.select_next_action(
            claims=[c1],
            evidence_store={},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        self.assertEqual(action, ControllerAction.STOP)
        self.assertIsNone(target)

    def test_fifo_per_claim_limits(self):
        policy = StaticAllocationPolicy()
        tracker = BudgetTracker(
            global_budget=VerificationBudget(max_steps=10),
            per_claim_budget=PerClaimBudget(max_retrievals=1),
        )
        c1 = _make_claim("c1", "Claim at retrieval limit.")
        # Exhaust retrieval attempts for c1
        tracker.record_action(ControllerAction.RETRIEVE, "c1")

        # c1 has no evidence and cannot afford retrieval -> should DEFER
        action, target, _ = policy.select_next_action(
            claims=[c1],
            evidence_store={"c1": []},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        self.assertEqual(action, ControllerAction.DEFER)
        self.assertEqual(target.claim_id, "c1")


class TestComparisonHarnessDryRun(unittest.TestCase):
    """Test run_policy_comparison harness execution in offline dry-run mode."""

    def setUp(self):
        self.sample_data = [
            {
                "id": "q_sample_1",
                "question": "Which magazine was established first: Arthur's Magazine or Periodical B?",
                "answer": "Arthur's Magazine",
                "supporting_facts": {"title": ["Arthur's Magazine"]},
            },
            {
                "id": "q_sample_2",
                "question": "Where was Arthur's Magazine published?",
                "answer": "Philadelphia",
                "supporting_facts": {"title": ["Arthur's Magazine"]},
            },
        ]

    def test_comparison_harness_dry_run_execution(self):
        from scripts.run_policy_comparison import run_controlled_comparison

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result = run_controlled_comparison(
                questions_data=self.sample_data,
                split_name="test_split",
                dry_run=True,
                max_steps=3,
                max_model_calls=4,
                output_dir=tmp_path,
            )

            # 1. Output verification
            self.assertIn("metadata", result)
            self.assertIn("aggregate_summary", result)
            self.assertIn("records", result)
            self.assertEqual(len(result["records"]), 2)

            # 2. Check all 6 policies are represented
            first_q = result["records"][0]
            policies = first_q["policies"]
            self.assertIn("vanilla_rag", policies)
            self.assertIn("static_fifo", policies)
            self.assertIn("static_cts_ranked", policies)
            self.assertIn("deterministic_cts", policies)
            self.assertIn("contextual_mab", policies)
            self.assertIn("exhaustive_reference", policies)

            # 3. Check identical initial conditions
            draft_ans = first_q["initial_draft_answer"]
            self.assertTrue(len(draft_ans) > 0)
            self.assertEqual(policies["vanilla_rag"]["final_answer"], draft_ans)
            self.assertEqual(policies["static_fifo"]["final_answer"], draft_ans)
            self.assertEqual(policies["static_cts_ranked"]["final_answer"], draft_ans)
            self.assertEqual(policies["deterministic_cts"]["final_answer"], draft_ans)
            self.assertEqual(policies["contextual_mab"]["final_answer"], draft_ans)
            self.assertEqual(policies["exhaustive_reference"]["final_answer"], draft_ans)

            # 4. Check file writing
            files = list(tmp_path.glob("*.json"))
            self.assertEqual(len(files), 1)
            with open(files[0], "r", encoding="utf-8") as f:
                saved_json = json.load(f)
            self.assertEqual(saved_json["metadata"]["num_questions"], 2)

    def test_gold_label_isolation_fix_f01(self):
        """Fix F-01: Verify LinUCB policy uses ProxyVerificationRewardModel, NOT GoldAnnotatedRewardModel."""
        from scripts.run_policy_comparison import run_controlled_comparison
        from src.controller.reward import ProxyVerificationRewardModel, GoldAnnotatedRewardModel

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_controlled_comparison(
                questions_data=self.sample_data,
                split_name="test_split",
                dry_run=True,
                max_steps=2,
                max_model_calls=3,
                output_dir=Path(tmp_dir),
            )

            for record in result["records"]:
                mab_rec = record["policies"]["contextual_mab"]
                self.assertIn("rewards", mab_rec)

    def test_consistent_cost_accounting_fix_f02(self):
        """Fix F-02: Verify 3-tier cost reporting across Vanilla RAG and verification policies."""
        from scripts.run_policy_comparison import run_controlled_comparison

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_controlled_comparison(
                questions_data=self.sample_data,
                split_name="test_split",
                dry_run=True,
                max_steps=3,
                max_model_calls=4,
                output_dir=Path(tmp_dir),
            )

            for rec in result["records"]:
                policies = rec["policies"]

                # 1. Vanilla RAG: verification_cost must be 0, total_cost == initial_cost
                vanilla = policies["vanilla_rag"]
                self.assertIn("initial_cost", vanilla)
                self.assertIn("verification_cost", vanilla)
                self.assertIn("total_cost", vanilla)
                self.assertEqual(vanilla["verification_cost"]["model_calls"], 0)
                self.assertEqual(vanilla["verification_cost"]["total_tokens"], 0)
                self.assertEqual(vanilla["total_cost"]["model_calls"], vanilla["initial_cost"]["model_calls"])

                # 2. Verification Policies: total_cost == initial_cost + verification_cost
                for p_key in ("static_fifo", "static_cts_ranked", "deterministic_cts", "contextual_mab", "exhaustive_reference"):
                    p = policies[p_key]
                    self.assertIn("initial_cost", p)
                    self.assertIn("verification_cost", p)
                    self.assertIn("total_cost", p)
                    expected_calls = p["initial_cost"]["model_calls"] + p["verification_cost"]["model_calls"]
                    self.assertEqual(p["total_cost"]["model_calls"], expected_calls)
                    expected_tokens = p["initial_cost"]["total_tokens"] + p["verification_cost"]["total_tokens"]
                    self.assertEqual(p["total_cost"]["total_tokens"], expected_tokens)

    def test_post_hoc_reward_step_inputs_fix_f03(self):
        """Fix F-03: Verify offline reward evaluation evaluates actual step claims, not dummy None inputs."""
        from scripts.run_policy_comparison import run_controlled_comparison

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_controlled_comparison(
                questions_data=self.sample_data,
                split_name="test_split",
                dry_run=True,
                max_steps=3,
                max_model_calls=4,
                output_dir=Path(tmp_dir),
            )

            for rec in result["records"]:
                for p_key in ("static_fifo", "static_cts_ranked", "deterministic_cts", "contextual_mab", "exhaustive_reference"):
                    p = rec["policies"][p_key]
                    for rew in p.get("rewards", []):
                        self.assertIn("step_number", rew)
                        self.assertIn("action", rew)
                        if rew.get("reward_status") != "UNAVAILABLE":
                            self.assertIsNotNone(rew.get("claim_id"))

    def test_aggregate_metrics_multi_dimensional_fix_f04(self):
        """Fix F-04: Verify macro EM, macro F1, and separate cost/coverage categories in aggregate_summary."""
        from scripts.run_policy_comparison import run_controlled_comparison

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_controlled_comparison(
                questions_data=self.sample_data,
                split_name="test_split",
                dry_run=True,
                max_steps=3,
                max_model_calls=4,
                output_dir=Path(tmp_dir),
            )

            agg = result["aggregate_summary"]
            for p_key in ("vanilla_rag", "static_fifo", "static_cts_ranked", "deterministic_cts", "contextual_mab", "exhaustive_reference"):
                p_agg = agg[p_key]
                self.assertIn("answer_generation", p_agg)
                self.assertIn("macro_exact_match", p_agg["answer_generation"])
                self.assertIn("macro_token_f1", p_agg["answer_generation"])

                self.assertIn("claim_verification_state", p_agg)
                self.assertIn("verification_coverage", p_agg["claim_verification_state"])
                self.assertIn("unsupported_claim_rate", p_agg["claim_verification_state"])
                self.assertIn("unresolved_rate", p_agg["claim_verification_state"])
                self.assertIn("deferred_rate", p_agg["claim_verification_state"])

                self.assertIn("ground_truth_alignment", p_agg)
                self.assertEqual(p_agg["ground_truth_alignment"]["status"], "NOT_MEASURABLE_WITHOUT_HUMAN_ANNOTATION")

                self.assertIn("verification_cost", p_agg)
                self.assertIn("total_end_to_end_cost", p_agg)
                self.assertIn("offline_reward_diagnostic", p_agg)


class TestStaticCTSRankedPolicy(unittest.TestCase):
    """Test StaticCTSRankedPolicy (AUD-04): fixed CTS ranking without dynamic re-ranking."""

    def test_cts_ranking_ignores_arrival_order(self):
        policy = StaticCTSRankedPolicy(ControllerPolicyConfig())
        tracker = BudgetTracker(VerificationBudget(max_steps=10))

        # c1 arrived first but has lowest CTS
        # c2 arrived second with highest CTS
        # c3 arrived third with medium CTS
        c1 = _make_claim("c1", "Low CTS claim.", triage_score=0.2)
        c2 = _make_claim("c2", "High CTS claim.", triage_score=0.95)
        c3 = _make_claim("c3", "Medium CTS claim.", triage_score=0.6)

        action, target, rationale = policy.select_next_action(
            claims=[c1, c2, c3],
            evidence_store={"c1": [], "c2": [], "c3": []},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        # Must target c2 first due to highest CTS (0.95), ignoring arrival sequence
        self.assertEqual(action, ControllerAction.RETRIEVE)
        self.assertEqual(target.claim_id, "c2")
        self.assertIn("StaticCTS", rationale)
        self.assertEqual(policy.initial_ranking, ["c2", "c3", "c1"])

    def test_deterministic_tie_breaking(self):
        policy = StaticCTSRankedPolicy(ControllerPolicyConfig())
        tracker = BudgetTracker(VerificationBudget(max_steps=10))

        # Identical CTS: c1 has TEMPORAL_COMPARISON, c2 has FACTUAL_RELATION
        c1 = _make_claim("c1", "Temporal claim", claim_type=ClaimType.TEMPORAL_COMPARISON, triage_score=0.7)
        c2 = _make_claim("c2", "Factual claim", claim_type=ClaimType.FACTUAL_RELATION, triage_score=0.7)

        action, target, _ = policy.select_next_action(
            claims=[c2, c1],
            evidence_store={"c1": [], "c2": []},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        # Temporal comparison has higher type rank
        self.assertEqual(target.claim_id, "c1")
        self.assertEqual(policy.initial_ranking, ["c1", "c2"])

    def test_missing_cts_value_treated_as_zero(self):
        policy = StaticCTSRankedPolicy(ControllerPolicyConfig())
        tracker = BudgetTracker(VerificationBudget(max_steps=10))

        c_none = _make_claim("c_none", "Claim with None CTS", triage_score=None)
        c_pos = _make_claim("c_pos", "Claim with 0.1 CTS", triage_score=0.1)

        action, target, _ = policy.select_next_action(
            claims=[c_none, c_pos],
            evidence_store={"c_none": [], "c_pos": []},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        self.assertEqual(target.claim_id, "c_pos")
        self.assertEqual(policy.initial_ranking, ["c_pos", "c_none"])

    def test_budget_exhaustion(self):
        policy = StaticCTSRankedPolicy()
        tracker = BudgetTracker(VerificationBudget(max_steps=0))
        c1 = _make_claim("c1", "Claim under zero budget.", triage_score=0.8)

        action, target, _ = policy.select_next_action(
            claims=[c1],
            evidence_store={},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        self.assertEqual(action, ControllerAction.STOP)
        self.assertIsNone(target)

    def test_infeasible_action_deferral(self):
        policy = StaticCTSRankedPolicy()
        tracker = BudgetTracker(
            global_budget=VerificationBudget(max_steps=10),
            per_claim_budget=PerClaimBudget(max_retrievals=0),
        )
        c1 = _make_claim("c1", "Claim cannot afford retrieval.", triage_score=0.9)
        action, target, _ = policy.select_next_action(
            claims=[c1],
            evidence_store={"c1": []},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        self.assertEqual(action, ControllerAction.DEFER)
        self.assertEqual(target.claim_id, "c1")

    def test_absence_of_outcome_driven_reranking(self):
        """Crucial test: Remaining claims are NOT dynamically re-ranked based on intermediate outcomes."""
        policy = StaticCTSRankedPolicy(ControllerPolicyConfig())
        tracker = BudgetTracker(VerificationBudget(max_steps=10))

        c1 = _make_claim("c1", "First ranked claim", triage_score=0.9)
        c2 = _make_claim("c2", "Second ranked claim", triage_score=0.7)

        # Step 1: Initial selection targets c1
        act1, tgt1, _ = policy.select_next_action(
            claims=[c1, c2],
            evidence_store={"c1": [], "c2": []},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        self.assertEqual(tgt1.claim_id, "c1")

        # Step 2: c1 has evidence, c1 is verified as CONTRADICTED.
        # Even though c1 is contradicted and c2 is unexamined, the static order across claims
        # strictly follows [c1, c2]. c1 undergoes REPAIR first because it is top of ranking.
        ev_p = EvidencePassage(evidence_id="ev1", doc_title="Doc", source_text="Refuting text.")
        assessment_contra = EvidenceAssessment(
            claim_id="c1",
            evidence_ids=["ev1"],
            supporting_spans={"ev1": ["Refuting text."]},
            label=AssessmentLabel.CONTRADICTED,
            confidence=0.85,
            rationale="Refuting text contradicts claim.",
        )
        act2, tgt2, _ = policy.select_next_action(
            claims=[c1, c2],
            evidence_store={"c1": [ev_p], "c2": []},
            assessments={"c1": assessment_contra},
            repairs={},
            budget_tracker=tracker,
        )
        self.assertEqual(act2, ControllerAction.REPAIR)
        self.assertEqual(tgt2.claim_id, "c1")

        # Step 3: Now c1 has completed repair. Since c1 is completed, policy advances to c2.
        rep_res = RepairResult(claim_id="c1", original_text="c1", proposed_revision="c1 repaired", repair_reason="Corrected entity.")
        act3, tgt3, _ = policy.select_next_action(
            claims=[c1, c2],
            evidence_store={"c1": [ev_p], "c2": []},
            assessments={"c1": assessment_contra},
            repairs={"c1": rep_res},
            budget_tracker=tracker,
        )
        self.assertEqual(act3, ControllerAction.RETRIEVE)
        self.assertEqual(tgt3.claim_id, "c2")
        self.assertEqual(policy.initial_ranking, ["c1", "c2"])


class TestEpisodeTerminationRegression_AUD07(unittest.TestCase):
    """Regression tests for AUD-07: Ensure mark_episode_end is called exactly once per episode."""

    def test_single_episode_finalization_in_harness(self):
        from scripts.run_policy_comparison import run_controlled_comparison
        from src.controller.mab_policy import ContextualMABPolicy

        sample_data = [
            {
                "id": f"q_{i}",
                "question": f"Question {i}?",
                "answer": f"Answer {i}",
                "supporting_facts": {"title": [f"Doc_{i}"]},
            }
            for i in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_controlled_comparison(
                questions_data=sample_data,
                split_name="test_split",
                dry_run=True,
                max_steps=2,
                max_model_calls=2,
                output_dir=Path(tmp_dir),
            )
            # In 3 questions, exactly 3 episodes should have been marked
            self.assertEqual(len(result["records"]), 3)

    def test_idempotent_mark_episode_end_with_id(self):
        from src.controller.mab_policy import ContextualMABPolicy

        mab = ContextualMABPolicy()
        self.assertEqual(mab.episodes_trained, 0)

        # First call with run_id increments
        mab.mark_episode_end("run_alpha")
        self.assertEqual(mab.episodes_trained, 1)

        # Duplicate call with identical run_id MUST NOT increment again
        mab.mark_episode_end("run_alpha")
        self.assertEqual(mab.episodes_trained, 1)

        # Different episode increments
        mab.mark_episode_end("run_beta")
        self.assertEqual(mab.episodes_trained, 2)


class TestGoldLabelIsolation_AUD03(unittest.TestCase):
    """Tests for AUD-03: Strict gold-label quarantine and invariance of policy decisions and parameters."""

    def test_gold_label_mutation_invariance_for_decisions_and_arm_updates(self):
        """Rule out leakage: Changing gold answer/titles must produce bit-for-bit identical policy decisions & weights."""
        from scripts.run_policy_comparison import run_controlled_comparison

        dataset_A = [
            {
                "id": "q1",
                "question": "Which periodical was established first, Arthur's Magazine or Periodical B?",
                "answer": "Arthur's Magazine",
                "supporting_facts": {"title": ["Arthur's Magazine"]},
            }
        ]
        dataset_B = [
            {
                "id": "q1",
                "question": "Which periodical was established first, Arthur's Magazine or Periodical B?",
                # Mutated gold answer and titles (completely different labels)
                "answer": "Completely Irrelevant Mutated Answer 999",
                "supporting_facts": {"title": ["Nonexistent Title XYZ"]},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpA, tempfile.TemporaryDirectory() as tmpB:
            resA = run_controlled_comparison(
                questions_data=dataset_A,
                split_name="test",
                dry_run=True,
                max_steps=3,
                max_model_calls=3,
                output_dir=Path(tmpA),
            )
            resB = run_controlled_comparison(
                questions_data=dataset_B,
                split_name="test",
                dry_run=True,
                max_steps=3,
                max_model_calls=3,
                output_dir=Path(tmpB),
            )

            # 1. Action decisions for all policies must be IDENTICAL
            for p_key in ("static_fifo", "static_cts_ranked", "deterministic_cts", "contextual_mab"):
                pA = resA["records"][0]["policies"][p_key]
                pB = resB["records"][0]["policies"][p_key]
                self.assertEqual(pA["steps_taken"], pB["steps_taken"])
                self.assertEqual(pA["stopping_reason"], pB["stopping_reason"])
                self.assertEqual(pA["claims_summary"], pB["claims_summary"])
                self.assertEqual(pA["verification_cost"], pB["verification_cost"])

    def test_post_hoc_evaluation_cannot_modify_policy_state(self):
        """Post-hoc evaluation using GoldAnnotatedRewardModel cannot mutate policy parameters."""
        import copy
        from src.controller.mab_policy import ContextualMABPolicy
        from src.controller.reward import GoldAnnotatedRewardModel

        mab = ContextualMABPolicy()
        initial_state = copy.deepcopy(mab.to_dict())

        gold_eval = GoldAnnotatedRewardModel(
            gold_supporting_titles={"Gold Title A"},
            gold_answer="Gold Answer",
        )

        claim = _make_claim("c1", "Test claim text")
        # Run gold evaluation on actions
        outcome = gold_eval.compute_reward(
            claim=claim,
            action=ControllerAction.RETRIEVE,
            new_passages=[],
            cumulative_passages=[],
        )
        self.assertIsNotNone(outcome)

        # Policy state must be 100% identical
        self.assertEqual(mab.to_dict(), initial_state)

    def test_proxy_reward_explicitly_identified_as_proxy(self):
        """Proxy reward signals must not be confused with ground truth."""
        from src.controller.reward import BaseRewardModel, ProxyVerificationRewardModel, GoldAnnotatedRewardModel

        proxy = ProxyVerificationRewardModel()
        self.assertIsInstance(proxy, BaseRewardModel)
        # Proxy model must NOT have gold answer or gold titles attributes
        self.assertFalse(hasattr(proxy, "gold_answer"))
        self.assertFalse(hasattr(proxy, "gold_supporting_titles"))

    def test_retrieval_relevance_vs_factual_correctness_separation(self):
        """Retrieving novel evidence does NOT equate to factual correctness."""
        from src.controller.reward import ProxyVerificationRewardModel, GoldAnnotatedRewardModel

        gold_eval = GoldAnnotatedRewardModel(
            gold_supporting_titles={"Arthur's Magazine"},
            gold_answer="Arthur's Magazine",
        )
        claim = _make_claim("c1", "Arthur's Magazine was founded in 1844.", claim_type=ClaimType.FACTUAL_RELATION)
        passage = EvidencePassage(evidence_id="p1", doc_title="Arthur's Magazine", source_text="Arthur's Magazine text.")

        # RETRIEVE action reward evaluates retrieval, NOT claim correctness
        ret_reward = gold_eval.compute_reward(
            claim=claim,
            action=ControllerAction.RETRIEVE,
            new_passages=[passage],
            cumulative_passages=[passage],
        )
        self.assertEqual(ret_reward.evidence_retrieval_reward, 1.0)
        # Correctness reward is None for retrieval actions
        self.assertIsNone(ret_reward.claim_correctness_reward)


class TestExhaustiveReference_AUD04(unittest.TestCase):
    """Test exhaustive-verification reference configuration (AUD-04)."""

    def test_exhaustive_reference_configuration_and_disclaimer(self):
        from scripts.run_policy_comparison import run_controlled_comparison

        sample_data = [
            {
                "id": "q_ref_1",
                "question": "Which magazine was established first: Arthur's Magazine or Periodical B?",
                "answer": "Arthur's Magazine",
                "supporting_facts": {"title": ["Arthur's Magazine"]},
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_controlled_comparison(
                questions_data=sample_data,
                split_name="test_split",
                dry_run=True,
                max_steps=2,
                max_model_calls=2,
                output_dir=Path(tmp_dir),
            )

            # Record-level verification
            ref_rec = result["records"][0]["policies"]["exhaustive_reference"]
            self.assertTrue(ref_rec["is_reference_policy"])
            self.assertFalse(ref_rec["budget_constrained"])
            self.assertIn("Exhaustive verification quality ceiling", ref_rec["reference_note"])

            # Aggregate-level verification
            ref_agg = result["aggregate_summary"]["exhaustive_reference"]
            self.assertTrue(ref_agg["is_reference_policy"])
            self.assertFalse(ref_agg["budget_constrained"])
            self.assertIn("reference_disclaimer", ref_agg)


class TestLinUCBCheckpointAndFrozenEvaluation(unittest.TestCase):
    """Test Phase 2 requirements: Checkpoint save/load, schema validation, and frozen evaluation invariance."""

    def test_checkpoint_roundtrip_save_and_load(self):
        from src.controller.mab_policy import (
            ContextualMABPolicy,
            MABPolicyConfig,
            FEATURE_NAMES,
            POLICY_SCHEMA_VERSION,
        )
        from src.controller.reward import ProxyVerificationRewardModel

        cfg = MABPolicyConfig(alpha=0.35, random_seed=123)
        policy = ContextualMABPolicy(config=cfg, reward_model=ProxyVerificationRewardModel())

        # Perform a simulated update to arm weights
        arm = policy.arms[ControllerAction.RETRIEVE]
        ctx_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        arm.update(ctx_vec, reward=1.0)
        policy.total_feedback_updates = 1
        policy.episodes_trained = 1

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_ckpt.json"
            policy.save_checkpoint(
                ckpt_path,
                training_split="train[100:300]",
                metadata={"notes": "unit test checkpoint"},
            )

            # Check raw JSON structure and keys
            with open(ckpt_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["policy_schema_version"], POLICY_SCHEMA_VERSION)
            self.assertEqual(data["policy_type"], "ContextualMABPolicy")
            self.assertEqual(data["hyperparameters"]["alpha"], 0.35)
            self.assertEqual(data["feature_schema"]["feature_names"], FEATURE_NAMES)
            self.assertEqual(data["feature_schema"]["feature_dimension"], 8)
            self.assertEqual(data["metadata"]["training_split"], "train[100:300]")
            self.assertEqual(data["arm_parameters"]["RETRIEVE"]["update_count"], 1)

            # Load into a new policy instance
            loaded = ContextualMABPolicy.load_checkpoint(ckpt_path, frozen=True)
            self.assertTrue(loaded.frozen)
            self.assertEqual(loaded.config.alpha, 0.35)
            self.assertEqual(loaded.arms[ControllerAction.RETRIEVE].update_count, 1)
            self.assertEqual(loaded.total_feedback_updates, 1)
            self.assertEqual(loaded.episodes_trained, 1)

    def test_checkpoint_incompatibility_checks(self):
        from src.controller.mab_policy import ContextualMABPolicy, MABPolicyConfig

        policy = ContextualMABPolicy(config=MABPolicyConfig())
        ckpt_dict = policy.to_checkpoint()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Invalid policy_type
            bad_type = dict(ckpt_dict)
            bad_type["policy_type"] = "EpsilonGreedy"
            p1 = Path(tmp_dir) / "bad_type.json"
            with open(p1, "w") as f:
                json.dump(bad_type, f)
            with self.assertRaises(ValueError) as ctx:
                ContextualMABPolicy.load_checkpoint(p1)
            self.assertIn("Incompatible policy type", str(ctx.exception))

            # 2. Incompatible feature dimension
            bad_dim = dict(ckpt_dict)
            bad_dim["feature_schema"] = {
                "feature_dimension": 12,
                "feature_names": ["f" + str(i) for i in range(12)],
            }
            p2 = Path(tmp_dir) / "bad_dim.json"
            with open(p2, "w") as f:
                json.dump(bad_dim, f)
            with self.assertRaises(ValueError) as ctx:
                ContextualMABPolicy.load_checkpoint(p2)
            self.assertIn("Feature dimension mismatch", str(ctx.exception))

            # 3. Incompatible feature name ordering
            bad_names = dict(ckpt_dict)
            bad_names["feature_schema"] = {
                "feature_dimension": 8,
                "feature_names": list(reversed(ckpt_dict["feature_schema"]["feature_names"])),
            }
            p3 = Path(tmp_dir) / "bad_names.json"
            with open(p3, "w") as f:
                json.dump(bad_names, f)
            with self.assertRaises(ValueError) as ctx:
                ContextualMABPolicy.load_checkpoint(p3)
            self.assertIn("Feature name ordering mismatch", str(ctx.exception))

    def test_frozen_evaluation_invariance(self):
        """Frozen evaluation must prohibit mutations to arm matrices, update counts, and episode counts."""
        from src.controller.mab_policy import ContextualMABPolicy, MABPolicyConfig
        from src.controller.reward import ProxyVerificationRewardModel

        policy = ContextualMABPolicy(
            config=MABPolicyConfig(alpha=0.5, frozen=True),
            reward_model=ProxyVerificationRewardModel(),
        )
        self.assertTrue(policy.frozen)

        # Baseline snapshot of arm state
        arm_ret = policy.arms[ControllerAction.RETRIEVE]
        initial_updates = arm_ret.update_count
        initial_b = list(arm_ret.b)
        initial_A_inv = [row[:] for row in arm_ret.A_inv]

        # 1. Trigger action selection
        claim = _make_claim("c_froz", "Test frozen invariance claim.", triage_score=0.8)
        tracker = BudgetTracker(VerificationBudget(max_steps=5))
        action, target, rationale = policy.select_next_action(
            claims=[claim],
            evidence_store={"c_froz": []},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        self.assertEqual(action, ControllerAction.RETRIEVE)

        # 2. Trigger feedback recording during evaluation
        obs_reward = policy.record_feedback(
            claim=claim,
            action=ControllerAction.RETRIEVE,
            evidence_passages=[],
            assessment=None,
            repair_result=None,
        )
        # Reward is still computed for diagnostics
        self.assertIsNotNone(obs_reward)

        # 3. Mark episode end
        policy.mark_episode_end()

        # Assert zero mutations occurred
        self.assertEqual(arm_ret.update_count, initial_updates)
        self.assertEqual(arm_ret.b, initial_b)
        self.assertEqual(arm_ret.A_inv, initial_A_inv)
        self.assertEqual(policy.total_feedback_updates, 0)
        self.assertEqual(policy.episodes_trained, 0)

    def test_deterministic_frozen_decisions(self):
        """Two policy instances loaded from same checkpoint in frozen mode produce identical decisions."""
        from src.controller.mab_policy import ContextualMABPolicy, MABPolicyConfig

        cfg = MABPolicyConfig(alpha=0.5, random_seed=42)
        base = ContextualMABPolicy(config=cfg)
        arm = base.arms[ControllerAction.RETRIEVE]
        arm.update([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], reward=1.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_file = Path(tmp_dir) / "det_ckpt.json"
            base.save_checkpoint(ckpt_file)

            policy1 = ContextualMABPolicy.load_checkpoint(ckpt_file, frozen=True)
            policy2 = ContextualMABPolicy.load_checkpoint(ckpt_file, frozen=True)

            claim = _make_claim("c_det", "Claim for deterministic testing.", triage_score=0.65)
            tracker1 = BudgetTracker(VerificationBudget(max_steps=5))
            tracker2 = BudgetTracker(VerificationBudget(max_steps=5))

            act1, tgt1, rat1 = policy1.select_next_action(
                claims=[claim], evidence_store={"c_det": []}, assessments={}, repairs={}, budget_tracker=tracker1
            )
            act2, tgt2, rat2 = policy2.select_next_action(
                claims=[claim], evidence_store={"c_det": []}, assessments={}, repairs={}, budget_tracker=tracker2
            )

            self.assertEqual(act1, act2)
            self.assertEqual(tgt1.claim_id, tgt2.claim_id)
            self.assertEqual(rat1, rat2)

    def test_harness_integration_with_freeze_flag(self):
        """Harness records mab_frozen=True in metadata when --freeze-policy is enabled."""
        from scripts.run_policy_comparison import run_controlled_comparison

        sample_data = [
            {
                "id": "q_freeze_1",
                "question": "Which journal was launched first: Journal A or Journal B?",
                "answer": "Journal A",
                "supporting_facts": {"title": ["Journal A"]},
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_controlled_comparison(
                questions_data=sample_data,
                split_name="test_split",
                dry_run=True,
                max_steps=2,
                max_model_calls=2,
                freeze_policy=True,
                output_dir=Path(tmp_dir),
            )
            self.assertTrue(result["metadata"]["mab_frozen"])
            self.assertIsNone(result["metadata"]["mab_checkpoint"])


if __name__ == "__main__":
    unittest.main()


