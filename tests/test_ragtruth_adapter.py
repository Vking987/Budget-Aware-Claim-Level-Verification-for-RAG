"""Unit and deterministic fixture tests for RAGTruth dataset adapter and claim mapping."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus
from src.evaluation.ragtruth_adapter import (
    ClaimAnnotationLabel,
    ClaimAnnotationMapper,
    MappingStatus,
    RAGTruthDatasetAdapter,
    RAGTruthLabelSpan,
    RAGTruthResponseRecord,
    RAGTruthSourceRecord,
    compute_claim_evaluation_metrics,
)


class TestRAGTruthAdapterAndMapping(unittest.TestCase):
    """Deterministic test suite validating the RAGTruth pipeline requirements."""

    def setUp(self) -> None:
        self.fixture_path = Path("tests/fixtures/ragtruth_sample.json")
        self.adapter = RAGTruthDatasetAdapter()
        self.adapter.load_from_fixture(self.fixture_path)
        self.mapper = ClaimAnnotationMapper()

    # 1. Split preservation and example-ID preservation
    def test_split_and_id_preservation(self):
        self.assertEqual(len(self.adapter.responses), 6)
        # Check IDs
        self.assertEqual(self.adapter.responses[0].response_id, "resp_001_clean")
        self.assertEqual(self.adapter.responses[0].source_id, "ragtruth_fixture_001")
        self.assertEqual(self.adapter.responses[0].split, "test")

        self.assertEqual(self.adapter.responses[2].response_id, "resp_003_multi_overlap")
        self.assertEqual(self.adapter.responses[2].split, "train")

    # 2. No annotation overlap -> UNFLAGGED (not asserted as verified support)
    def test_no_annotation_overlap(self):
        resp_clean = self.adapter.responses[0]
        mapped = self.mapper.map_response(resp_clean)
        self.assertGreaterEqual(len(mapped), 1)
        for c in mapped:
            self.assertEqual(c.mapped_label, ClaimAnnotationLabel.UNFLAGGED)
            self.assertEqual(c.mapping_status, MappingStatus.CLEAN_UNANNOTATED)
            self.assertEqual(len(c.mapped_span_ids), 0)

    # 3. One hallucinated span fully inside one claim
    def test_hallucinated_span_fully_inside_claim(self):
        resp_hall = self.adapter.responses[1]  # "Arthur's Magazine was published in Boston. It was edited by T.S. Arthur."
        # Label: "Boston" (35 to 41)
        mapped = self.mapper.map_response(resp_hall)
        self.assertEqual(len(mapped), 2)
        # First claim contains "Boston"
        claim_boston = mapped[0]
        self.assertIn("Boston", claim_boston.claim_text)
        self.assertEqual(claim_boston.mapped_label, ClaimAnnotationLabel.HALLUCINATED)
        self.assertEqual(claim_boston.mapping_status, MappingStatus.CONTAINED_SPAN)
        self.assertIn("span_0", claim_boston.mapped_span_ids)

        # Second claim: "It was edited by T.S. Arthur." has no annotation -> UNFLAGGED
        claim_editor = mapped[1]
        self.assertEqual(claim_editor.mapped_label, ClaimAnnotationLabel.UNFLAGGED)

    # 4. A claim with both annotated and unannotated spans (partial overlap)
    def test_claim_with_annotated_and_unannotated_spans(self):
        resp_hall = self.adapter.responses[1]
        mapped = self.mapper.map_response(resp_hall)
        claim_boston = mapped[0]
        # Offset character length of claim vs annotation
        self.assertGreater(len(claim_boston.claim_text), len("Boston"))
        self.assertEqual(claim_boston.mapped_label, ClaimAnnotationLabel.HALLUCINATED)

    # 5. One annotation overlapping multiple claims -> Marked UNCERTAIN
    def test_annotation_overlapping_multiple_claims_is_uncertain(self):
        resp_multi = self.adapter.responses[2]
        # "Aulani is located in Honolulu. Djuan Rivers managed the luxury property there."
        # Label: "Honolulu. Djuan Rivers" (crosses sentence/claim boundary)
        mapped = self.mapper.map_response(resp_multi)
        self.assertGreaterEqual(len(mapped), 2)
        # Both overlapping claims should reflect multi_claim_span or uncertain
        labels = [c.mapped_label for c in mapped]
        self.assertTrue(any(lbl == ClaimAnnotationLabel.UNCERTAIN for lbl in labels))
        statuses = [c.mapping_status for c in mapped]
        self.assertTrue(any(st == MappingStatus.MULTI_CLAIM_SPAN for st in statuses))

    # 6. Punctuation and Unicode offsets
    def test_punctuation_and_unicode_offsets(self):
        resp_unicode = self.adapter.responses[3]
        # "The café’s journal was established in 1999—an exciting year."
        # Label: "1999" (start 38, end 42)
        # Verify text at offsets matches exactly
        self.assertEqual(resp_unicode.response[38:42], "1999")
        mapped = self.mapper.map_response(resp_unicode)
        self.assertGreaterEqual(len(mapped), 1)
        found_hall = False
        for c in mapped:
            if "1999" in c.claim_text:
                self.assertEqual(c.mapped_label, ClaimAnnotationLabel.HALLUCINATED)
                found_hall = True
        self.assertTrue(found_hall)

    # 7. Missing or malformed annotations (out of bounds or text mismatch) -> Marked UNCERTAIN
    def test_malformed_annotations_marked_uncertain(self):
        resp_malformed = self.adapter.responses[4]
        mapped = self.mapper.map_response(resp_malformed)
        self.assertGreaterEqual(len(mapped), 1)
        # Never silently converted into unflagged or hallucinated claim
        for c in mapped:
            self.assertEqual(c.mapped_label, ClaimAnnotationLabel.UNCERTAIN)
            self.assertIn(c.mapping_status, (MappingStatus.OUT_OF_BOUNDS, MappingStatus.MALFORMED_SPAN))

    # 8. Handling of implicit_true: factually true external world knowledge kept distinct
    def test_implicit_true_handling(self):
        resp_implicit = self.adapter.responses[5]
        # "The capital of France is Paris." with label "Paris" (implicit_true=True)
        mapped = self.mapper.map_response(resp_implicit)
        self.assertEqual(len(mapped), 1)
        claim_paris = mapped[0]
        # Must be IMPLICIT_TRUE_ABSENT, not HALLUCINATED and not UNFLAGGED
        self.assertEqual(claim_paris.mapped_label, ClaimAnnotationLabel.IMPLICIT_TRUE_ABSENT)
        self.assertIn("span_0", claim_paris.mapped_span_ids)

    # 9. Correct exclusion of uncertain and implicit_true examples from supervised metrics
    def test_exclusion_of_uncertain_and_implicit_true_from_supervised_metrics(self):
        all_mapped = self.adapter.process_all()
        metrics = compute_claim_evaluation_metrics(all_mapped, self.adapter.responses, cts_threshold=0.50)

        # Check total vs supervised
        self.assertGreater(metrics.total_claims, metrics.supervised_claim_count)
        self.assertEqual(
            metrics.supervised_claim_count,
            metrics.unflagged_count + metrics.hallucinated_count
        )
        self.assertGreater(metrics.uncertain_count, 0)
        self.assertGreater(metrics.implicit_true_absent_count, 0)
        self.assertEqual(
            metrics.total_claims,
            metrics.supervised_claim_count + metrics.uncertain_count + metrics.mixed_count + metrics.implicit_true_absent_count
        )
        # Denominators must be transparently reported
        self.assertIn("supervised_claims", metrics.denominators)
        self.assertIn("excluded_uncertain_mixed", metrics.denominators)
        self.assertEqual(
            metrics.denominators["excluded_uncertain_mixed"],
            metrics.uncertain_count + metrics.mixed_count + metrics.implicit_true_absent_count
        )
