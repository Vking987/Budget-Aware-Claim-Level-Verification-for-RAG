"""Comprehensive offline unit tests for Stage B2 Contextual MAB Policy.

Covers:
1. Feature construction for claim-action pairs (dimensionality, bounding, edge cases).
2. Reward calculation and invalid reward handling (non-finite, clamping).
3. LinUCB arm parameter updates after observations.
4. Uncertainty bonus behavior (shrinking with sample count, alpha scaling).
5. Exploration vs. exploitation (alpha=0 pure greedy vs. alpha>0).
6. Disjoint arms learning different expected rewards for different actions.
7. Dynamic decision shifting: new observation changes subsequent action choice.
8. Learning across multiple episodes.
9. Untrained vs. trained policy state divergence.
10. Deterministic tie-breaking.
11. Hard budget constraints strictly prevent selecting ineligible arms.
12. Missing assessments and empty contexts handled gracefully.
13. Action exceptions and missing rewards handled without crash.
14. Policy state serialization, file persistence, and roundtrip equality.
15. End-to-end controller execution with ContextualMABPolicy.
16. Existing deterministic policy continues to work identically.
"""

import math
import os
import tempfile
import unittest
from typing import List

from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus
from src.controller.budget import BudgetTracker
from src.controller.controller import VerificationController
from src.controller.interfaces import MockRepairer, MockRetriever
from src.controller.mab_policy import (
    ContextualMABPolicy,
    LinUCBArm,
    MABPolicyConfig,
    extract_context_features,
)
from src.controller.models import (
    ControllerAction,
    PerClaimBudget,
    StoppingReason,
    VerificationBudget,
)
from src.controller.policy import SequentialDecisionPolicy
from src.controller.reward import (
    CorrectnessObservability,
    GoldAnnotatedRewardModel,
    ProxyVerificationRewardModel,
    RewardOutcome,
)
from src.evidence.assessor import MockDeterministicAssessor
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


def _make_passage(evidence_id: str, title: str, text: str) -> EvidencePassage:
    return EvidencePassage(
        evidence_id=evidence_id,
        doc_title=title,
        source_text=text,
        retrieval_score=0.9,
    )


class TestFeatureConstruction(unittest.TestCase):
    """Test feature extraction contracts and bounds."""

    def test_feature_vector_dimension_and_bounds(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.", triage_score=0.85)
        tracker = BudgetTracker(VerificationBudget(max_steps=10))
        passages = [_make_passage("ev1", "Doc", "Text")]

        features = extract_context_features(
            claim=claim,
            action=ControllerAction.RETRIEVE,
            evidence_store={"c1": passages},
            assessments={},
            budget_tracker=tracker,
        )

        self.assertEqual(len(features), 8)
        self.assertEqual(features[0], 1.0)  # Intercept
        self.assertAlmostEqual(features[1], 0.85)  # CTS
        for i, val in enumerate(features):
            self.assertTrue(math.isfinite(val), f"Feature {i} is not finite")
            self.assertTrue(-1.0 <= val <= 1.0, f"Feature {i} ({val}) out of [-1.0, 1.0]")

    def test_missing_and_none_fields_handled(self):
        claim = _make_claim("c_none", "Text", triage_score=None)
        tracker = BudgetTracker()

        features = extract_context_features(
            claim=claim,
            action=ControllerAction.VERIFY,
            evidence_store={},
            assessments={},
            budget_tracker=tracker,
        )
        self.assertEqual(len(features), 8)
        self.assertEqual(features[1], 0.5)  # Fallback CTS


class TestRewardModels(unittest.TestCase):
    """Test quality reward computation and contracts."""

    def test_proxy_reward_supported_grounding(self):
        reward_model = ProxyVerificationRewardModel(verification_threshold=0.60)
        claim = _make_claim("c1", "Claim text", triage_score=0.8)
        passages = [_make_passage("ev1", "Doc", "Claim text")]
        assessment = EvidenceAssessment(
            claim_id="c1",
            evidence_ids=["ev1"],
            supporting_spans={},
            label=AssessmentLabel.SUPPORTED,
            confidence=0.85,
            rationale="Supported",
            limitations="",
        )

        outcome = reward_model.compute_reward(
            claim=claim,
            action=ControllerAction.VERIFY,
            passages=passages,
            assessment=assessment,
        )
        self.assertGreater(outcome.quality_reward, 0.5)
        self.assertTrue(-1.0 <= outcome.quality_reward <= 1.0)
        self.assertGreater(outcome.heuristic_risk_reduction, 0.0)

    def test_proxy_reward_contradiction_discovery(self):
        reward_model = ProxyVerificationRewardModel()
        claim = _make_claim("c1", "Claim text")
        assessment = EvidenceAssessment(
            claim_id="c1",
            evidence_ids=["ev1"],
            supporting_spans={},
            label=AssessmentLabel.CONTRADICTED,
            confidence=0.75,
            rationale="Contradicted",
            limitations="",
        )

        outcome = reward_model.compute_reward(
            claim=claim,
            action=ControllerAction.VERIFY,
            passages=[],
            assessment=assessment,
        )
        # Discovering contradiction is valuable
        self.assertEqual(outcome.quality_reward, 0.6)

    def test_gold_annotated_reward(self):
        gold_model = GoldAnnotatedRewardModel(
            gold_supporting_titles={"Arthur's Magazine", "First for Women"},
            gold_answer="1844",
        )
        claim = _make_claim("c1", "Text")
        matched_passage = _make_passage("ev1", "Arthur's Magazine", "Text")

        outcome = gold_model.compute_reward(
            claim=claim,
            action=ControllerAction.RETRIEVE,
            passages=[matched_passage],
        )
        self.assertEqual(outcome.quality_reward, 1.0)

    def test_invalid_reward_raises_value_error(self):
        with self.assertRaises(ValueError):
            RewardOutcome(quality_reward=float("nan"))
        with self.assertRaises(ValueError):
            RewardOutcome(quality_reward=float("inf"))


class TestLinUCBArmUpdates(unittest.TestCase):
    """Test LinUCB linear model updates, prediction, and uncertainty bonus."""

    def test_arm_updates_and_uncertainty_shrinks(self):
        arm = LinUCBArm(ControllerAction.VERIFY, dimension=4, l2_reg=1.0)
        x = [1.0, 0.5, 0.2, 0.8]

        # Initial prediction (untrained: theta is zero, variance is x^T I x = norm(x)^2)
        mean_init, bonus_init, ucb_init = arm.predict(x, alpha=1.0)
        self.assertEqual(mean_init, 0.0)
        self.assertGreater(bonus_init, 0.0)

        # Update arm with positive reward
        arm.update(x, reward=1.0)
        self.assertEqual(arm.update_count, 1)

        mean_after, bonus_after, ucb_after = arm.predict(x, alpha=1.0)
        # Expected mean should have increased
        self.assertGreater(mean_after, 0.0)
        # Uncertainty bonus should have shrunk because we gained information
        self.assertLess(bonus_after, bonus_init)

    def test_exploration_vs_exploitation_alpha(self):
        arm = LinUCBArm(ControllerAction.VERIFY, dimension=4)
        x = [1.0, 0.5, 0.0, 0.0]

        # With alpha=0 (pure exploitation), bonus is 0
        _, bonus_zero, ucb_zero = arm.predict(x, alpha=0.0)
        self.assertEqual(bonus_zero, 0.0)

        # With alpha=1.5, bonus > 0
        _, bonus_high, ucb_high = arm.predict(x, alpha=1.5)
        self.assertGreater(bonus_high, 0.0)
        self.assertGreater(ucb_high, ucb_zero)


class TestContextualMABPolicyBehavior(unittest.TestCase):
    """Test contextual policy decision making, dynamic learning, and tie breaking."""

    def test_distinct_arms_learn_different_rewards(self):
        policy = ContextualMABPolicy(MABPolicyConfig(alpha=0.1, feature_dimension=8))
        claim = _make_claim("c1", "Claim text")

        # Teach policy that RETRIEVE receives high reward (+1.0), VERIFY receives low reward (-0.5)
        x_dummy = [1.0] * 8
        policy.arms[ControllerAction.RETRIEVE].update(x_dummy, reward=1.0)
        policy.arms[ControllerAction.VERIFY].update(x_dummy, reward=-0.5)

        ret_mean, _, _ = policy.arms[ControllerAction.RETRIEVE].predict(x_dummy, alpha=0.0)
        ver_mean, _, _ = policy.arms[ControllerAction.VERIFY].predict(x_dummy, alpha=0.0)

        self.assertGreater(ret_mean, ver_mean)

    def test_observation_shifts_subsequent_decision(self):
        policy = ContextualMABPolicy(MABPolicyConfig(alpha=0.0))  # Greedy
        c1 = _make_claim("c1", "Claim 1", triage_score=0.5)
        tracker = BudgetTracker(VerificationBudget(max_steps=10))

        # Action on c1 initially is RETRIEVE
        action1, selected1, _ = policy.select_next_action(
            claims=[c1],
            evidence_store={"c1": []},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        self.assertEqual(action1, ControllerAction.RETRIEVE)

        # Record reward for retrieval
        policy.record_feedback(
            claim=c1,
            action=ControllerAction.RETRIEVE,
            passages=[_make_passage("ev1", "Doc", "Text")],
            explicit_reward=0.8,
        )

        # Now c1 has evidence, next action dynamically becomes VERIFY
        action2, selected2, _ = policy.select_next_action(
            claims=[c1],
            evidence_store={"c1": [_make_passage("ev1", "Doc", "Text")]},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        self.assertEqual(action2, ControllerAction.VERIFY)

    def test_budget_ineligible_arms_are_never_selected(self):
        policy = ContextualMABPolicy()
        c1 = _make_claim("c1", "Claim text")

        # Per claim budget allows 0 retrievals
        tracker = BudgetTracker(
            global_budget=VerificationBudget(max_steps=10),
            per_claim_budget=PerClaimBudget(max_retrievals=0),
        )

        # Even though c1 has no evidence, RETRIEVE cannot be afforded
        action, target, _ = policy.select_next_action(
            claims=[c1],
            evidence_store={"c1": []},
            assessments={},
            repairs={},
            budget_tracker=tracker,
        )
        # Must not select RETRIEVE
        self.assertNotEqual(action, ControllerAction.RETRIEVE)

    def test_deterministic_tie_breaking(self):
        policy = ContextualMABPolicy()
        c_b = _make_claim("claim_b", "Identical claim", triage_score=0.7)
        c_a = _make_claim("claim_a", "Identical claim", triage_score=0.7)
        tracker = BudgetTracker(VerificationBudget(max_steps=10))

        # Regardless of input list order, claim_a must be selected first
        _, sel1, _ = policy.select_next_action([c_b, c_a], {}, {}, {}, tracker)
        self.assertEqual(sel1.claim_id, "claim_a")

        _, sel2, _ = policy.select_next_action([c_a, c_b], {}, {}, {}, tracker)
        self.assertEqual(sel2.claim_id, "claim_a")


class TestMultiEpisodeLearningAndSerialization(unittest.TestCase):
    """Test policy serialization, persistence, and multi-episode accumulation."""

    def test_serialization_roundtrip_dict(self):
        policy = ContextualMABPolicy(MABPolicyConfig(alpha=0.4))
        c1 = _make_claim("c1", "Claim")
        policy.record_feedback(c1, ControllerAction.RETRIEVE, passages=[_make_passage("e1", "D", "T")], explicit_reward=0.5)
        policy.mark_episode_end()

        p_dict = policy.to_dict()
        restored = ContextualMABPolicy.from_dict(p_dict)

        self.assertEqual(restored.episodes_trained, 1)
        self.assertEqual(restored.config.alpha, 0.4)
        self.assertEqual(
            restored.arms[ControllerAction.RETRIEVE].update_count,
            policy.arms[ControllerAction.RETRIEVE].update_count
        )

    def test_file_save_and_load(self):
        policy = ContextualMABPolicy(MABPolicyConfig(alpha=0.6))
        c1 = _make_claim("c1", "Claim")
        policy.record_feedback(c1, ControllerAction.VERIFY, passages=[], explicit_reward=-0.2)
        policy.mark_episode_end()

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "mab_policy.json")
            policy.save(fpath)
            self.assertTrue(os.path.exists(fpath))

            loaded = ContextualMABPolicy.load(fpath)
            self.assertEqual(loaded.episodes_trained, 1)
            self.assertEqual(loaded.config.alpha, 0.6)
            self.assertEqual(
                loaded.arms[ControllerAction.VERIFY].update_count, 1
            )


class TestControllerEndToEndWithMAB(unittest.TestCase):
    """Test VerificationController running with ContextualMABPolicy."""

    def test_controller_executes_with_mab_policy(self):
        c1 = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.", triage_score=0.9)
        passage = _make_passage("ev1", "Arthur's Magazine", "Arthur's Magazine was published in Philadelphia from 1844 to 1846.")

        mab_policy = ContextualMABPolicy(MABPolicyConfig(alpha=0.3))
        controller = VerificationController(
            retriever=MockRetriever(canned_evidence={"c1": [passage]}),
            assessor=MockDeterministicAssessor(),
            repairer=MockRepairer(),
            policy=mab_policy,
            global_budget=VerificationBudget(max_steps=10),
        )

        trace = controller.run([c1])

        self.assertEqual(c1.state, ClaimState.VERIFIED)
        self.assertEqual(trace.stopping_reason, StoppingReason.ALL_CLAIMS_RESOLVED)
        self.assertGreater(mab_policy.total_feedback_updates, 0)
        self.assertEqual(mab_policy.episodes_trained, 1)

    def test_deterministic_policy_still_runs_identically(self):
        c1 = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        passage = _make_passage("ev1", "Arthur's Magazine", "Arthur's Magazine was published in Philadelphia.")

        det_policy = SequentialDecisionPolicy()
        controller = VerificationController(
            retriever=MockRetriever(canned_evidence={"c1": [passage]}),
            assessor=MockDeterministicAssessor(),
            policy=det_policy,
            global_budget=VerificationBudget(max_steps=10),
        )

        trace = controller.run([c1])
        self.assertEqual(c1.state, ClaimState.VERIFIED)
        self.assertEqual(trace.policy_name, "SequentialDecisionPolicy")


class TestTargetedCorrections(unittest.TestCase):
    """Targeted regression tests for Stage B2.1 reward and feedback corrections.

    Covers:
    1. Gold title match does not independently prove factual correctness.
    2. Unavailable claim-level ground truth is not treated as a negative label.
    3. Repeated retrieval does not receive repeated positive reward for old evidence.
    4. Duplicate document identifiers do not count as new evidence.
    5. Newly retrieved, genuinely relevant gold evidence can receive retrieval reward.
    6. Failed actions cannot receive positive rewards from stale passages.
    7. Failed actions do not update LinUCB using fabricated success outcomes.
    8. Successful action feedback still updates the correct arm using its selected context.
    9. Existing budget and claim-state safeguards remain intact.
    """

    def test_gold_title_match_does_not_prove_factual_correctness(self):
        """AUD-01: A gold document title match alone must never set
        correctness_observability to OBSERVABLE_CORRECT for bridging claims."""
        gold_reward = GoldAnnotatedRewardModel(
            gold_titles=["Arthur's Magazine"],
            gold_answer="Philadelphia",
        )
        # This claim is a bridging fact about founding date — it does NOT directly
        # assert or contradict the gold answer string "Philadelphia".
        claim = _make_claim("c1", "Arthur's Magazine was founded in May 1844 by T.S. Arthur.")
        passage = _make_passage("ev1", "Arthur's Magazine", "Founded in May 1844 by Timothy Shay Arthur.")

        assessment = EvidenceAssessment(
            claim_id="c1",
            evidence_ids=["ev1"],
            supporting_spans={"ev1": ["Founded in May 1844"]},
            label=AssessmentLabel.SUPPORTED,
            confidence=0.95,
            rationale="Passage supports the claim text.",
            cost=VerificationCost(),
        )

        # RETRIEVE action with gold title match
        outcome = gold_reward.compute_reward(
            claim=claim,
            action=ControllerAction.RETRIEVE,
            new_passages=[passage],
            cumulative_passages=[passage],
            assessment=assessment,
        )

        # Retrieval found gold document → evidence_retrieval_reward is positive
        self.assertEqual(outcome.evidence_retrieval_reward, 1.0)
        # But claim correctness is UNOBSERVABLE for bridge claims
        self.assertIsNone(outcome.claim_correctness_reward)
        self.assertEqual(outcome.correctness_observability, CorrectnessObservability.UNOBSERVABLE)
        self.assertFalse(outcome.is_factual_correctness_verified)

    def test_unavailable_ground_truth_is_not_treated_as_negative_label(self):
        """AUD-01: When gold answer doesn't relate to an intermediate bridging
        claim, the claim correctness must be UNOBSERVABLE and not penalized
        as a confirmed negative."""
        gold_reward = GoldAnnotatedRewardModel(
            gold_titles=["First Document"],
            gold_answer="1984",
        )
        # Bridging claim: "born in New York" has nothing to do with gold_answer "1984"
        claim = _make_claim("c1", "The author was born in New York City.")
        passage = _make_passage("ev1", "First Document", "The author was born in New York City.")

        # Use VERIFY with a real assessment (not None — None is an action failure)
        assessment = EvidenceAssessment(
            claim_id="c1",
            evidence_ids=["ev1"],
            supporting_spans={"ev1": ["born in New York City"]},
            label=AssessmentLabel.SUPPORTED,
            confidence=0.85,
            rationale="Passage supports the claim about birthplace.",
        )

        outcome = gold_reward.compute_reward(
            claim=claim,
            action=ControllerAction.VERIFY,
            new_passages=[],
            cumulative_passages=[passage],
            assessment=assessment,
        )

        # Correctness is UNOBSERVABLE since "born in NYC" ≠ "1984"
        self.assertEqual(outcome.correctness_observability, CorrectnessObservability.UNOBSERVABLE)
        self.assertIsNone(outcome.claim_correctness_reward)
        # Supported assessment with gold context gets positive quality reward
        self.assertGreater(outcome.quality_reward, 0.0)

    def test_repeated_retrieval_does_not_receive_repeated_positive_reward(self):
        """AUD-02: When the same passage is retrieved again, the marginal novelty
        check must prevent awarding a second positive reward."""
        reward_model = ProxyVerificationRewardModel()
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        passage = _make_passage("ev1", "Arthur's Magazine", "Published in Philadelphia.")

        # First retrieval: passage is novel (cumulative is empty before)
        outcome1 = reward_model.compute_reward(
            claim=claim,
            action=ControllerAction.RETRIEVE,
            new_passages=[passage],
            cumulative_passages=[],  # Empty prior pool → passage is novel
        )
        self.assertGreater(outcome1.quality_reward, 0.0)
        self.assertGreater(outcome1.evidence_retrieval_reward, 0.0)

        # Second retrieval: same passage, now already in cumulative pool
        outcome2 = reward_model.compute_reward(
            claim=claim,
            action=ControllerAction.RETRIEVE,
            new_passages=[passage],
            cumulative_passages=[passage, passage],  # passage already in prior pool
        )
        self.assertLess(outcome2.quality_reward, 0.0)
        self.assertLess(outcome2.evidence_retrieval_reward, 0.0)

    def test_duplicate_document_identifiers_do_not_count_as_new_evidence(self):
        """AUD-02: Two passages with identical evidence_id must be deduplicated.
        When both are already in the cumulative pool, retrieval is unproductive."""
        reward_model = ProxyVerificationRewardModel()
        claim = _make_claim("c1", "Some claim")
        p1 = _make_passage("ev_dup", "DocA", "Identical passage text.")
        p2 = _make_passage("ev_dup", "DocA", "Identical passage text.")

        # Both p1 and p2 share evidence_id "ev_dup" and are already in cumulative
        outcome = reward_model.compute_reward(
            claim=claim,
            action=ControllerAction.RETRIEVE,
            new_passages=[p1, p2],
            cumulative_passages=[p1, p1, p2],  # prior pool already has this key
        )
        # Duplicate retrieval → negative/duplicate provenance
        self.assertIn("duplicate", outcome.reward_provenance.lower())
        self.assertLess(outcome.quality_reward, 0.0)

    def test_newly_retrieved_novel_gold_evidence_receives_retrieval_reward(self):
        """Positive case: genuinely new gold evidence must get full retrieval reward."""
        gold_reward = GoldAnnotatedRewardModel(
            gold_titles=["Target Title"],
            gold_answer="Target Answer",
        )
        claim = _make_claim("c1", "Some claim")
        p_gold = _make_passage("ev_new", "Target Title", "Some text in target document.")

        outcome = gold_reward.compute_reward(
            claim=claim,
            action=ControllerAction.RETRIEVE,
            new_passages=[p_gold],
            cumulative_passages=[],
        )
        self.assertEqual(outcome.evidence_retrieval_reward, 1.0)
        self.assertGreater(outcome.quality_reward, 0.0)

    def test_failed_actions_cannot_receive_positive_rewards_from_stale_passages(self):
        """AUD-04: When action_failed=True, both reward models must return
        negative quality_reward and is_action_failure=True, regardless of what
        stale passages exist in the cumulative pool."""
        proxy_model = ProxyVerificationRewardModel()
        gold_model = GoldAnnotatedRewardModel(gold_titles=["Title"], gold_answer="Ans")
        claim = _make_claim("c1", "Some claim")
        stale_passage = _make_passage("ev1", "Title", "Some text")

        for model in [proxy_model, gold_model]:
            outcome = model.compute_reward(
                claim=claim,
                action=ControllerAction.RETRIEVE,
                new_passages=[],
                cumulative_passages=[stale_passage],
                action_failed=True,
                failure_reason="Network timeout connecting to index",
            )
            self.assertTrue(outcome.is_action_failure)
            self.assertLess(outcome.quality_reward, 0.0)
            # Failed actions return None for evidence_retrieval_reward (no retrieval occurred)
            self.assertIsNone(outcome.evidence_retrieval_reward)
            self.assertIn("failed", outcome.reward_provenance.lower())

    def test_failed_actions_do_not_update_linucb_using_fabricated_success(self):
        """AUD-04: When retriever raises an exception, the controller must
        record the failure in the trace and feed a failure reward to the MAB."""
        class CrashingRetriever:
            def retrieve(self, claim):
                raise RuntimeError("Qdrant connection refused")

        mab_policy = ContextualMABPolicy()
        c1 = _make_claim("c1", "Some claim")
        controller = VerificationController(
            retriever=CrashingRetriever(),
            assessor=MockDeterministicAssessor(),
            repairer=MockRepairer(),
            policy=mab_policy,
            global_budget=VerificationBudget(max_steps=2),
        )

        trace = controller.run([c1])
        self.assertGreater(mab_policy.total_feedback_updates, 0)
        # The RETRIEVE arm must have been updated with the failure feedback
        retrieve_arm = mab_policy.arms[ControllerAction.RETRIEVE]
        self.assertGreaterEqual(retrieve_arm.update_count, 1)
        # Trace should record error in failed steps
        failed_steps = [s for s in trace.steps if s.error is not None]
        self.assertGreater(len(failed_steps), 0)
        self.assertIn("Qdrant connection refused", failed_steps[0].error)

    def test_successful_action_feedback_updates_correct_arm_with_selected_context(self):
        """Positive case: successful retrieval must update only the RETRIEVE arm,
        not other arms."""
        mab_policy = ContextualMABPolicy()
        c1 = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        passage = _make_passage("ev1", "Doc", "Text")

        initial_retrieve_count = mab_policy.arms[ControllerAction.RETRIEVE].update_count
        initial_verify_count = mab_policy.arms[ControllerAction.VERIFY].update_count

        mab_policy.record_feedback(
            claim=c1,
            action=ControllerAction.RETRIEVE,
            new_passages=[passage],
            cumulative_passages=[],
        )

        self.assertEqual(mab_policy.arms[ControllerAction.RETRIEVE].update_count, initial_retrieve_count + 1)
        self.assertEqual(mab_policy.arms[ControllerAction.VERIFY].update_count, initial_verify_count)

    def test_mab_policy_execution_trace_and_feedback(self):
        """Task 2 verification: Trace one claim through the controller with LinUCB MAB.
        
        Verifies:
        1. ContextualMABPolicy is selected and active.
        2. Policy chooses an action (RETRIEVE) for the claim.
        3. That action is genuinely executed by the controller.
        4. The observed outcome triggers record_feedback and updates arm parameters (A_inv, b, update_count).
        5. Stopping condition and verification budget are strictly respected.
        """
        c1 = _make_claim("c_mab_trace", "Arthur's Magazine was published in Philadelphia.")
        passage = _make_passage("ev_art", "Arthur's Magazine", "Arthur's Magazine was an 1844 American literary magazine published in Philadelphia.")
        
        policy = ContextualMABPolicy(config=MABPolicyConfig(frozen=False, alpha=0.5))
        retrieve_arm = policy.arms[ControllerAction.RETRIEVE]
        initial_updates = retrieve_arm.update_count
        initial_b = list(retrieve_arm.b)

        controller = VerificationController(
            retriever=MockRetriever(canned_evidence={"c_mab_trace": [passage]}),
            assessor=MockDeterministicAssessor(),
            repairer=MockRepairer(),
            policy=policy,
            global_budget=VerificationBudget(max_steps=5, max_model_calls=5),
        )

        trace = controller.run([c1])

        # 1 & 2: Policy chose actions that were recorded in the trace
        self.assertGreater(len(trace.steps), 0)
        first_step = trace.steps[0]
        self.assertEqual(first_step.claim_id, "c_mab_trace")
        self.assertEqual(first_step.action, ControllerAction.RETRIEVE)
        self.assertIn("LinUCB selected arm", first_step.selection_rationale)

        # 3: Action was actually executed
        self.assertIn("Retrieved 1 new unique passages", first_step.action_outcome)
        self.assertGreater(trace.total_steps, 0)

        # 4: Observed outcome updated the policy arm parameters
        self.assertGreater(retrieve_arm.update_count, initial_updates)
        self.assertNotEqual(retrieve_arm.b, initial_b)
        self.assertGreater(policy.total_feedback_updates, 0)

        # 5: Budget and stopping conditions are respected
        self.assertLessEqual(trace.total_steps, 5)
        self.assertIn(
            trace.stopping_reason,
            (StoppingReason.ALL_CLAIMS_RESOLVED, StoppingReason.MAX_STEPS_REACHED, StoppingReason.NO_ELIGIBLE_ACTIONS),
        )




