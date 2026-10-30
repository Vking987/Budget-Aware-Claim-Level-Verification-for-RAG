"""Regression tests for RAGTruth controller-evaluation integration and metric calculations.

Verifies:
1. Root cause reproduction: Stale pre-controller claims report 0 checked claims and 0.0 coverage,
   while post-controller claims report non-zero completed-verification coverage matching controller_execution.
2. Operational state quantities partition:
   claims_verified + claims_unresolved + claims_deferred + claims_selected + claims_pending == total_claims.
3. Separation of verification_coverage (completed VERIFIED / total) from verification_attempt_rate
   (selected for verification / total).
4. Annotation-based hallucination rates only computed when relevant denominator is nonzero.
5. Annotation-only evaluation retains existing behavior.
6. Repaired claims transition to UNRESOLVED and are NOT counted as VERIFIED.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus
from src.controller.interfaces import MockRepairer, MockRetriever
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost
from src.evaluation.ragtruth_adapter import (
    ClaimAnnotationLabel,
    ClaimAnnotationMapper,
    MappingStatus,
    MappedClaimAnnotation,
    RAGTruthDatasetAdapter,
    RAGTruthLabelSpan,
    RAGTruthResponseRecord,
    RAGTruthSourceRecord,
    compute_claim_evaluation_metrics,
    run_ragtruth_controller_evaluation,
)


class TestRAGTruthControllerMetricsRegression(unittest.TestCase):
    """Test suite for controller-evaluation integration and metric correctness."""

    def setUp(self) -> None:
        self.fixture_path = Path("tests/fixtures/ragtruth_sample.json")
        self.adapter = RAGTruthDatasetAdapter()
        self.adapter.load_from_fixture(self.fixture_path)

    def test_reproduce_and_fix_controller_metric_discrepancy(self):
        """Demonstrate that stale claims caused verification_coverage=0, and verify post-controller claims fix it."""
        # 1. Stale / pre-controller claims: all claims in PENDING
        mapped_claims = self.adapter.process_all()
        pre_metrics = compute_claim_evaluation_metrics(mapped_claims, self.adapter.responses)

        # Pre-execution: 0 verified, 0 checked, 0.0 coverage
        self.assertEqual(pre_metrics.verification_coverage, 0.0)
        self.assertEqual(pre_metrics.denominators["checked_claims"], 0)
        self.assertEqual(pre_metrics.claims_verified_count, 0)
        self.assertEqual(pre_metrics.claims_pending_count, pre_metrics.total_claims)
        self.assertIsNone(pre_metrics.hallucinated_rate_among_checked)

        # 2. Execute controller on mapped_claims in-place
        summary = run_ragtruth_controller_evaluation(
            adapter=self.adapter,
            mapped_claims=mapped_claims,
            policy_name="ContextualMAB_LinUCB",
            dry_run=True,
        )

        # Post-execution metrics computed on the mutated claims
        post_metrics = compute_claim_evaluation_metrics(mapped_claims, self.adapter.responses)

        # Discrepancy is resolved:
        # Verified claims in controller_execution must match metrics.claims_verified_count
        expected_verified = summary["verification_outcomes"]["VERIFIED"]
        self.assertGreater(expected_verified, 0)
        self.assertEqual(post_metrics.claims_verified_count, expected_verified)
        self.assertEqual(post_metrics.denominators["checked_claims"], expected_verified)
        self.assertEqual(post_metrics.denominators["verified_claims"], expected_verified)

        # Completed verification coverage must be non-zero and equal to VERIFIED / total_claims
        expected_coverage = expected_verified / post_metrics.total_claims
        self.assertAlmostEqual(post_metrics.verification_coverage, expected_coverage, places=4)
        self.assertGreater(post_metrics.verification_coverage, 0.0)

    def test_operational_quantities_partition_and_attempt_rate(self):
        """Operational state counts must partition total_claims exactly, and attempt_rate >= coverage."""
        mapped_claims = self.adapter.process_all()
        summary = run_ragtruth_controller_evaluation(
            adapter=self.adapter,
            mapped_claims=mapped_claims,
            policy_name="ContextualMAB_LinUCB",
            dry_run=True,
        )
        metrics = compute_claim_evaluation_metrics(mapped_claims, self.adapter.responses)

        # Sum of mutually exclusive states must equal total_claims
        state_sum = (
            metrics.claims_verified_count
            + metrics.claims_unresolved_count
            + metrics.claims_deferred_count
            + metrics.claims_selected_count
            + metrics.claims_pending_count
        )
        self.assertEqual(state_sum, metrics.total_claims)

        # Denominators must mirror operational quantities
        self.assertEqual(metrics.denominators["verified_claims"], metrics.claims_verified_count)
        self.assertEqual(metrics.denominators["unresolved_claims"], metrics.claims_unresolved_count)
        self.assertEqual(metrics.denominators["deferred_claims"], metrics.claims_deferred_count)
        self.assertEqual(metrics.denominators["selected_claims"], metrics.claims_selected_count)
        self.assertEqual(metrics.denominators["pending_claims"], metrics.claims_pending_count)

        # Verification attempt rate represents claims where verification was attempted
        self.assertEqual(
            metrics.claims_selected_for_verification,
            metrics.claims_verified_count + metrics.claims_unresolved_count + metrics.claims_selected_count
        )
        self.assertGreaterEqual(metrics.verification_attempt_rate, metrics.verification_coverage)
        self.assertAlmostEqual(
            metrics.verification_attempt_rate,
            metrics.claims_selected_for_verification / metrics.total_claims,
            places=4,
        )

    def test_annotation_only_evaluation_unchanged(self):
        """Annotation-only evaluation (no controller) must retain 0.0 coverage and None rates."""
        mapped_claims = self.adapter.process_all()
        metrics = compute_claim_evaluation_metrics(mapped_claims, self.adapter.responses)

        self.assertEqual(metrics.verification_coverage, 0.0)
        self.assertEqual(metrics.verification_attempt_rate, 0.0)
        self.assertEqual(metrics.abstention_rate, 0.0)
        self.assertEqual(metrics.claims_verified_count, 0)
        self.assertEqual(metrics.claims_unresolved_count, 0)
        self.assertEqual(metrics.claims_deferred_count, 0)
        self.assertEqual(metrics.claims_selected_count, 0)
        self.assertEqual(metrics.claims_pending_count, metrics.total_claims)
        self.assertEqual(metrics.claims_selected_for_verification, 0)
        self.assertEqual(metrics.denominators["checked_claims"], 0)
        self.assertEqual(metrics.denominators["deferred_claims"], 0)
        self.assertIsNone(metrics.hallucinated_rate_among_checked)
        self.assertIsNone(metrics.hallucinated_rate_among_deferred)

    def test_hallucination_rate_nonzero_denominator_guard(self):
        """Annotation-based hallucination rates must only be computed when relevant denominator > 0."""
        # Create controlled synthetic claims
        c1 = ClaimRecord(
            claim_id="c1",
            source_sentence_id="s1",
            original_text="Verified unflagged fact.",
            normalized_text="Verified unflagged fact.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
        )
        c2 = ClaimRecord(
            claim_id="c2",
            source_sentence_id="s1",
            original_text="Verified hallucinated claim.",
            normalized_text="Verified hallucinated claim.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.VERIFIED,
        )
        c3 = ClaimRecord(
            claim_id="c3",
            source_sentence_id="s1",
            original_text="Deferred unflagged claim.",
            normalized_text="Deferred unflagged claim.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.DEFERRED,
        )

        m1 = MappedClaimAnnotation(
            example_id="e1",
            response_id="r1",
            claim_id="c1",
            claim_text="Verified unflagged fact.",
            char_start=0,
            char_end=24,
            mapped_label=ClaimAnnotationLabel.UNFLAGGED,
            mapping_status=MappingStatus.CLEAN_UNANNOTATED,
            mapping_confidence=1.0,
            claim_record=c1,
        )
        m2 = MappedClaimAnnotation(
            example_id="e1",
            response_id="r1",
            claim_id="c2",
            claim_text="Verified hallucinated claim.",
            char_start=25,
            char_end=53,
            mapped_label=ClaimAnnotationLabel.HALLUCINATED,
            mapping_status=MappingStatus.CONTAINED_SPAN,
            mapping_confidence=1.0,
            claim_record=c2,
        )
        m3 = MappedClaimAnnotation(
            example_id="e1",
            response_id="r1",
            claim_id="c3",
            claim_text="Deferred unflagged claim.",
            char_start=54,
            char_end=79,
            mapped_label=ClaimAnnotationLabel.UNFLAGGED,
            mapping_status=MappingStatus.CLEAN_UNANNOTATED,
            mapping_confidence=1.0,
            claim_record=c3,
        )

        resp = RAGTruthResponseRecord(
            response_id="r1",
            source_id="e1",
            model="test_model",
            temperature=0.0,
            split="test",
            quality="good",
            response="Verified unflagged fact. Verified hallucinated claim. Deferred unflagged claim.",
        )

        metrics = compute_claim_evaluation_metrics([m1, m2, m3], [resp])

        # 2 verified supervised claims, 1 of which is hallucinated -> 1/2 = 0.5
        self.assertEqual(metrics.claims_verified_count, 2)
        self.assertAlmostEqual(metrics.hallucinated_rate_among_checked, 0.5)
        self.assertAlmostEqual(metrics.hallucinated_rate_among_verified, 0.5)

        # 1 deferred supervised claim, 0 hallucinated -> 0/1 = 0.0
        self.assertEqual(metrics.claims_deferred_count, 1)
        self.assertAlmostEqual(metrics.hallucinated_rate_among_deferred, 0.0)

    def test_repaired_claim_transitions_to_unresolved_not_verified(self):
        """A repaired claim transitions to UNRESOLVED; it must NOT be counted as VERIFIED."""
        claim = ClaimRecord(
            claim_id="c_rep",
            source_sentence_id="s1",
            original_text="Contradicted statement.",
            normalized_text="Contradicted statement.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
            state=ClaimState.SELECTED,
        )

        # Simulate repair outcome: transitions to UNRESOLVED
        claim.transition_to(ClaimState.UNRESOLVED, reason="Claim qualified/repaired: Qualified text.")
        self.assertEqual(claim.state, ClaimState.UNRESOLVED)

        m = MappedClaimAnnotation(
            example_id="e1",
            response_id="r1",
            claim_id="c_rep",
            claim_text="Contradicted statement.",
            char_start=0,
            char_end=23,
            mapped_label=ClaimAnnotationLabel.HALLUCINATED,
            mapping_status=MappingStatus.CONTAINED_SPAN,
            mapping_confidence=1.0,
            claim_record=claim,
        )
        resp = RAGTruthResponseRecord(
            response_id="r1",
            source_id="e1",
            model="m",
            temperature=0.0,
            split="test",
            quality="good",
            response="Contradicted statement.",
        )

        metrics = compute_claim_evaluation_metrics([m], [resp])
        # Must be in unresolved, NOT in verified
        self.assertEqual(metrics.claims_verified_count, 0)
        self.assertEqual(metrics.claims_unresolved_count, 1)
        self.assertEqual(metrics.verification_coverage, 0.0)
