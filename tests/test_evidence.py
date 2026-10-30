"""Comprehensive tests for evidence alignment module.

All tests use the MockDeterministicAssessor and hand-checked fixtures.
No running LLM server is required.

Test coverage:
- Direct support (content overlap)
- Direct contradiction (negation mismatch)
- Irrelevant evidence
- Missing/empty evidence
- Conflicting passages
- Negation preservation
- Temporal comparison handling
- Numerical claim handling
- Multiple passages jointly supporting a claim
- Ambiguous evidence
- Missing or malformed evidence metadata
- Stable claim and evidence provenance tracing
- Retrieval score alone cannot produce SUPPORTED
- Assessment label is separate from ClaimState.VERIFIED
- LlamaAssessor response parsing (with mocked client)
- LlamaAssessor error handling
"""

import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.claims.models import (
    ClaimRecord,
    ClaimState,
    ClaimType,
    DecompositionStatus,
    InvalidStateTransitionError,
)
from src.evidence.models import (
    AssessmentLabel,
    EvidenceAssessment,
    EvidencePassage,
    VerificationCost,
)
from src.evidence.assessor import MockDeterministicAssessor
from src.evidence.llama_assessor import LlamaAssessor, _parse_assessment_response


# ==============================================================================
# Test Fixtures
# ==============================================================================

def _make_claim(
    claim_id: str,
    text: str,
    claim_type: ClaimType = ClaimType.FACTUAL_RELATION,
) -> ClaimRecord:
    """Create a ClaimRecord fixture for testing."""
    return ClaimRecord(
        claim_id=claim_id,
        source_sentence_id="s_test",
        original_text=text,
        normalized_text=text,
        claim_type=claim_type,
        decomposition_status=DecompositionStatus.ATOMIC,
    )


def _make_passage(
    evidence_id: str,
    title: str,
    text: str,
    score: float = None,
) -> EvidencePassage:
    """Create an EvidencePassage fixture for testing."""
    return EvidencePassage(
        evidence_id=evidence_id,
        doc_title=title,
        source_text=text,
        retrieval_score=score,
    )


class TestDirectSupport(unittest.TestCase):
    """Test that direct, content-bearing evidence correctly produces SUPPORTED."""

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_factual_claim_with_matching_evidence(self):
        claim = _make_claim("c1", "Arthur's Magazine was published in Philadelphia.")
        evidence = [
            _make_passage(
                "ev1", "Arthur's Magazine",
                "Arthur's Magazine was an American literary magazine published in "
                "Philadelphia from 1844 to 1846.",
                score=0.92
            )
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.SUPPORTED)
        self.assertEqual(result.claim_id, "c1")
        self.assertIn("ev1", result.evidence_ids)
        self.assertTrue(0.0 <= result.confidence <= 1.0)
        self.assertIsNotNone(result.rationale)
        self.assertIsNotNone(result.cost)

    def test_numerical_claim_with_exact_match(self):
        claim = _make_claim(
            "c2",
            "First for Women had a circulation of 1,310,696 copies.",
            claim_type=ClaimType.NUMERICAL_QUANTITATIVE,
        )
        evidence = [
            _make_passage(
                "ev2", "First for Women",
                "First for Women is a magazine that had a circulation of 1,310,696 copies "
                "as of the first half of 2009.",
            )
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.SUPPORTED)

    def test_temporal_claim_with_matching_years(self):
        claim = _make_claim(
            "c3",
            "Arthur's Magazine was started in 1844.",
            claim_type=ClaimType.TEMPORAL_COMPARISON,
        )
        evidence = [
            _make_passage(
                "ev3", "Arthur's Magazine",
                "Arthur's Magazine was founded in 1844 by Timothy Shay Arthur in Philadelphia.",
            )
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.SUPPORTED)


class TestDirectContradiction(unittest.TestCase):
    """Test that directly contradicting evidence produces CONTRADICTED."""

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_negation_mismatch_contradiction(self):
        claim = _make_claim(
            "c4",
            "Arthur's Magazine was published in Philadelphia.",
        )
        evidence = [
            _make_passage(
                "ev4", "Arthur's Magazine",
                "Arthur's Magazine was not published in Philadelphia. "
                "The magazine was based in New York.",
            )
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.CONTRADICTED)
        self.assertIn("ev4", result.evidence_ids)

    def test_numerical_contradiction(self):
        claim = _make_claim(
            "c5",
            "The circulation of the magazine was 500,000 copies.",
            claim_type=ClaimType.NUMERICAL_QUANTITATIVE,
        )
        evidence = [
            _make_passage(
                "ev5", "Magazine Stats",
                "The circulation of the magazine was 1,310,696 copies in 2009.",
            )
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.CONTRADICTED)


class TestIrrelevantEvidence(unittest.TestCase):
    """Test that topically unrelated evidence produces INSUFFICIENT."""

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_completely_unrelated_evidence(self):
        claim = _make_claim("c6", "Arthur's Magazine was published in Philadelphia.")
        evidence = [
            _make_passage(
                "ev6", "Quantum Physics",
                "Quantum mechanics is a fundamental theory in physics that provides "
                "a description of the physical properties of nature at the scale of "
                "atoms and subatomic particles.",
                score=0.95  # High retrieval score, but irrelevant content
            )
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.INSUFFICIENT)

    def test_high_retrieval_score_does_not_establish_support(self):
        """Critical test: high retrieval similarity must NOT produce SUPPORTED
        when evidence content does not actually entail the claim."""
        claim = _make_claim("c7", "Peggy Seeger married James Miller.")
        evidence = [
            _make_passage(
                "ev7", "Peggy Seeger",
                "Peggy Seeger is a well-known folk singer from the United States. "
                "She has performed at many festivals worldwide.",
                score=0.98  # Very high retrieval score
            )
        ]
        result = self.assessor.assess(claim, evidence)
        # Must NOT be SUPPORTED just because retrieval score is high
        self.assertNotEqual(result.label, AssessmentLabel.SUPPORTED)


class TestMissingEvidence(unittest.TestCase):
    """Test handling of absent, empty, or malformed evidence."""

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_no_evidence_at_all(self):
        claim = _make_claim("c8", "Arthur's Magazine was published in Philadelphia.")
        result = self.assessor.assess(claim, [])
        self.assertEqual(result.label, AssessmentLabel.INSUFFICIENT)
        self.assertEqual(result.evidence_ids, [])
        self.assertGreater(result.confidence, 0.5)

    def test_empty_passage_text(self):
        claim = _make_claim("c9", "Arthur's Magazine was published in Philadelphia.")
        evidence = [
            _make_passage("ev9", "Arthur's Magazine", "", score=0.85)
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.INSUFFICIENT)

    def test_whitespace_only_passage(self):
        claim = _make_claim("c10", "First for Women was started in 1989.")
        evidence = [
            _make_passage("ev10", "Blank", "   \n\t  ", score=0.5)
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.INSUFFICIENT)

    def test_missing_metadata_fields(self):
        """Evidence with None retrieval score and metadata should still be processed."""
        claim = _make_claim("c11", "Arthur's Magazine was published in Philadelphia.")
        evidence = [
            EvidencePassage(
                evidence_id="ev11",
                doc_title="",  # Empty title
                source_text="Arthur's Magazine was published in Philadelphia in 1844.",
                retrieval_score=None,  # No score
                metadata=None,  # No metadata
            )
        ]
        result = self.assessor.assess(claim, evidence)
        # Should still assess based on content
        self.assertIn(result.label, [AssessmentLabel.SUPPORTED, AssessmentLabel.INSUFFICIENT])
        self.assertEqual(result.evidence_ids, ["ev11"])


class TestConflictingPassages(unittest.TestCase):
    """Test detection of conflicting evidence from multiple passages."""

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_one_supports_one_contradicts(self):
        claim = _make_claim("c12", "Arthur's Magazine was published in Philadelphia.")
        evidence = [
            _make_passage(
                "ev12a", "Source A",
                "Arthur's Magazine was published in Philadelphia from 1844 to 1846.",
            ),
            _make_passage(
                "ev12b", "Source B",
                "Arthur's Magazine was not published in Philadelphia. "
                "The magazine was based in Boston.",
            ),
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.CONFLICTED)
        self.assertIn("ev12a", result.evidence_ids)
        self.assertIn("ev12b", result.evidence_ids)
        self.assertIsNotNone(result.limitations)


class TestNegationPreservation(unittest.TestCase):
    """Test that negation in claims is handled correctly."""

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_negated_claim_supported_by_negated_evidence(self):
        """Claim with negation should be SUPPORTED when evidence also negates."""
        claim = _make_claim(
            "c13",
            "Neither Arthur's Magazine nor First for Women was started first.",
            claim_type=ClaimType.EXISTENCE_NEGATION,
        )
        evidence = [
            _make_passage(
                "ev13", "Comparison",
                "Neither Arthur's Magazine nor First for Women was started first, "
                "as they were published in different centuries.",
            )
        ]
        result = self.assessor.assess(claim, evidence)
        # Both contain negation and share entities → support, not contradiction
        self.assertNotEqual(result.label, AssessmentLabel.CONTRADICTED)

    def test_affirmative_claim_contradicted_by_negation(self):
        claim = _make_claim("c14", "Peggy Seeger married James Miller.")
        evidence = [
            _make_passage(
                "ev14", "Peggy Seeger",
                "Peggy Seeger never married James Miller. She was "
                "in a long-term partnership with Ewan MacColl.",
            )
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.CONTRADICTED)


class TestTemporalComparisons(unittest.TestCase):
    """Test handling of temporal ordering claims."""

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_temporal_ordering_supported(self):
        claim = _make_claim(
            "c15",
            "Arthur's Magazine was started in 1844 before First for Women in 1989.",
            claim_type=ClaimType.TEMPORAL_COMPARISON,
        )
        evidence = [
            _make_passage(
                "ev15", "Publication History",
                "Arthur's Magazine was founded in 1844. First for Women magazine "
                "was started in 1989, over a century later.",
            )
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.SUPPORTED)


class TestNumericalClaims(unittest.TestCase):
    """Test handling of quantitative and numerical claims."""

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_exact_number_match_supports(self):
        claim = _make_claim(
            "c16",
            "First for Women had a circulation of 1,310,696 copies.",
            claim_type=ClaimType.NUMERICAL_QUANTITATIVE,
        )
        evidence = [
            _make_passage(
                "ev16", "First for Women",
                "First for Women is a magazine with a circulation of 1,310,696 copies.",
            )
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.SUPPORTED)


class TestMultiplePassageJointSupport(unittest.TestCase):
    """Test that multiple passages can jointly support a claim."""

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_two_passages_provide_complementary_support(self):
        claim = _make_claim(
            "c17",
            "Arthur's Magazine was published in Philadelphia and was edited by T.S. Arthur.",
        )
        evidence = [
            _make_passage(
                "ev17a", "Arthur's Magazine",
                "Arthur's Magazine was an American literary magazine published in Philadelphia.",
            ),
            _make_passage(
                "ev17b", "T.S. Arthur",
                "Timothy Shay Arthur was the editor of Arthur's Magazine.",
            ),
        ]
        result = self.assessor.assess(claim, evidence)
        # At least one passage should provide enough overlap
        self.assertIn(result.label, [AssessmentLabel.SUPPORTED, AssessmentLabel.INSUFFICIENT])
        self.assertEqual(len(result.evidence_ids), 2)


class TestAmbiguousEvidence(unittest.TestCase):
    """Test that ambiguous or partial evidence produces INSUFFICIENT."""

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_tangentially_related_evidence(self):
        claim = _make_claim(
            "c18",
            "Peggy Seeger married James Miller in 1977.",
        )
        evidence = [
            _make_passage(
                "ev18", "Peggy Seeger",
                "Peggy Seeger is an American-born folk singer and political activist. "
                "She has lived in England for many years.",
                score=0.82
            )
        ]
        result = self.assessor.assess(claim, evidence)
        # Evidence mentions Peggy Seeger but says nothing about marriage
        self.assertNotEqual(result.label, AssessmentLabel.SUPPORTED)


class TestProvenanceAndTraceability(unittest.TestCase):
    """Test that assessment outputs maintain full traceability."""

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_claim_id_preserved_in_assessment(self):
        claim = _make_claim("trace_claim_42", "Arthur's Magazine was published in Philadelphia.")
        evidence = [
            _make_passage(
                "trace_ev_99", "Arthur's Magazine",
                "Arthur's Magazine was published in Philadelphia.",
            )
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.claim_id, "trace_claim_42")
        self.assertIn("trace_ev_99", result.evidence_ids)

    def test_all_evidence_ids_recorded(self):
        claim = _make_claim("c19", "Some claim text.")
        evidence = [
            _make_passage("ev_a", "Doc A", "Relevant text about the claim topic."),
            _make_passage("ev_b", "Doc B", "More text about the claim."),
            _make_passage("ev_c", "Doc C", "Additional context and information."),
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(set(result.evidence_ids), {"ev_a", "ev_b", "ev_c"})

    def test_assessor_id_recorded(self):
        claim = _make_claim("c20", "Test claim.")
        result = self.assessor.assess(claim, [])
        self.assertEqual(result.assessor_id, "mock_deterministic_v1")

    def test_serialization_roundtrip(self):
        claim = _make_claim("c21", "Arthur's Magazine was published in Philadelphia.")
        evidence = [
            _make_passage(
                "ev21", "Arthur's Magazine",
                "Arthur's Magazine was published in Philadelphia.",
            )
        ]
        result = self.assessor.assess(claim, evidence)
        d = result.to_dict()

        # Verify all required fields are present
        required_keys = {
            "claim_id", "evidence_ids", "supporting_spans", "label",
            "confidence", "confidence_note", "rationale", "limitations",
            "cost", "timestamp", "assessor_id",
        }
        self.assertTrue(required_keys.issubset(set(d.keys())))
        self.assertEqual(d["confidence_note"], "Model estimate, NOT calibrated probability.")
        self.assertIsInstance(d["cost"], dict)

    def test_verification_cost_tracked(self):
        claim = _make_claim("c22", "Test claim.")
        evidence = [
            _make_passage("ev22", "Doc", "Some evidence text about the test claim topic."),
        ]
        result = self.assessor.assess(claim, evidence)
        self.assertIsNotNone(result.cost)
        self.assertEqual(result.cost.model_calls, 0)  # Mock uses no LLM
        self.assertIsNotNone(result.cost.latency_seconds)
        self.assertGreaterEqual(result.cost.latency_seconds, 0.0)


class TestAssessmentDoesNotEquateToVerification(unittest.TestCase):
    """Critical test: an assessment label of SUPPORTED does NOT automatically
    transition a claim to ClaimState.VERIFIED.

    This tests the architectural boundary between the evidence assessor
    and the claim lifecycle state machine.
    """

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_supported_assessment_does_not_change_claim_state(self):
        claim = _make_claim("c23", "Arthur's Magazine was published in Philadelphia.")
        evidence = [
            _make_passage(
                "ev23", "Arthur's Magazine",
                "Arthur's Magazine was published in Philadelphia from 1844.",
            )
        ]

        # Assessment says SUPPORTED
        result = self.assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.SUPPORTED)

        # But claim state must still be PENDING — assessor doesn't touch it
        self.assertEqual(claim.state, ClaimState.PENDING)

    def test_pending_claim_cannot_jump_to_verified(self):
        """Even with a SUPPORTED assessment, a PENDING claim cannot jump
        directly to VERIFIED. It must go through SELECTED first."""
        claim = _make_claim("c24", "Test claim.")

        with self.assertRaises(InvalidStateTransitionError):
            claim.transition_to(
                ClaimState.VERIFIED,
                reason="Assessment said SUPPORTED",
                evidence={"assessment": "SUPPORTED"},
            )

    def test_selected_claim_can_be_verified_with_evidence(self):
        """Only a SELECTED claim with proper evidence payload can reach VERIFIED."""
        claim = _make_claim("c25", "Arthur's Magazine was published in Philadelphia.")

        # Proper lifecycle: PENDING → SELECTED → VERIFIED
        claim.transition_to(ClaimState.SELECTED, reason="Controller selected for verification")
        claim.transition_to(
            ClaimState.VERIFIED,
            reason="Evidence assessment confirmed support",
            evidence={
                "assessment_label": "SUPPORTED",
                "evidence_ids": ["ev25"],
                "assessor_id": "mock_deterministic_v1",
            }
        )
        self.assertEqual(claim.state, ClaimState.VERIFIED)


class TestModelConfidenceDisclaimer(unittest.TestCase):
    """Test that confidence is clearly marked as a model estimate."""

    def setUp(self):
        self.assessor = MockDeterministicAssessor()

    def test_serialized_output_includes_confidence_note(self):
        claim = _make_claim("c26", "Test claim.")
        result = self.assessor.assess(claim, [])
        d = result.to_dict()
        self.assertIn("confidence_note", d)
        self.assertIn("NOT calibrated probability", d["confidence_note"])

    def test_confidence_bounds_enforced(self):
        with self.assertRaises(ValueError):
            EvidenceAssessment(
                claim_id="bad",
                evidence_ids=[],
                supporting_spans={},
                label=AssessmentLabel.INSUFFICIENT,
                confidence=1.5,  # Out of bounds
                rationale="test",
            )

        with self.assertRaises(ValueError):
            EvidenceAssessment(
                claim_id="bad",
                evidence_ids=[],
                supporting_spans={},
                label=AssessmentLabel.INSUFFICIENT,
                confidence=-0.1,  # Out of bounds
                rationale="test",
            )


# ==============================================================================
# LlamaAssessor Tests (with mocked client)
# ==============================================================================

class TestLlamaAssessorResponseParsing(unittest.TestCase):
    """Test the JSON response parser used by LlamaAssessor."""

    def test_valid_json_response(self):
        raw = '{"label": "SUPPORTED", "confidence": 0.85, "rationale": "Evidence matches.", "supporting_spans": ["span1"], "limitations": null}'
        label, conf, rat, spans, lim = _parse_assessment_response(raw, "c1", ["ev1"])
        self.assertEqual(label, AssessmentLabel.SUPPORTED)
        self.assertAlmostEqual(conf, 0.85)
        self.assertEqual(rat, "Evidence matches.")

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"label": "CONTRADICTED", "confidence": 0.7, "rationale": "Negation mismatch.", "supporting_spans": [], "limitations": "Lexical only."}\n```'
        label, conf, rat, spans, lim = _parse_assessment_response(raw, "c2", ["ev2"])
        self.assertEqual(label, AssessmentLabel.CONTRADICTED)
        self.assertEqual(lim, "Lexical only.")

    def test_malformed_json_falls_back_to_insufficient(self):
        raw = "This is not valid JSON at all."
        label, conf, rat, spans, lim = _parse_assessment_response(raw, "c3", ["ev3"])
        self.assertEqual(label, AssessmentLabel.INSUFFICIENT)
        self.assertLess(conf, 0.5)
        self.assertIn("Failed to parse", rat)

    def test_unknown_label_falls_back_to_insufficient(self):
        raw = '{"label": "MAYBE_TRUE", "confidence": 0.5, "rationale": "Uncertain.", "supporting_spans": [], "limitations": null}'
        label, conf, rat, spans, lim = _parse_assessment_response(raw, "c4", ["ev4"])
        self.assertEqual(label, AssessmentLabel.INSUFFICIENT)

    def test_out_of_range_confidence_clamped(self):
        raw = '{"label": "SUPPORTED", "confidence": 1.5, "rationale": "Very sure.", "supporting_spans": [], "limitations": null}'
        label, conf, rat, spans, lim = _parse_assessment_response(raw, "c5", ["ev5"])
        self.assertLessEqual(conf, 1.0)

        raw2 = '{"label": "SUPPORTED", "confidence": -0.3, "rationale": "Negative.", "supporting_spans": [], "limitations": null}'
        label2, conf2, rat2, spans2, lim2 = _parse_assessment_response(raw2, "c6", ["ev6"])
        self.assertGreaterEqual(conf2, 0.0)

    def test_unclosed_json_recovery(self):
        # Model produced valid JSON fields but omitted trailing closing brace
        raw = '{\n  "label": "INSUFFICIENT",\n  "confidence": 0.5,\n  "rationale": "Partial context."\n'
        label, conf, rat, spans, lim = _parse_assessment_response(raw, "c7", ["ev7"])
        self.assertEqual(label, AssessmentLabel.INSUFFICIENT)
        self.assertAlmostEqual(conf, 0.5)
        self.assertEqual(rat, "Partial context.")

    def test_embedded_json_recovery(self):
        # Model returned preamble/postamble around JSON
        raw = 'Here is the evaluation:\n{"label": "SUPPORTED", "confidence": 0.9, "rationale": "Clear match."}\nHope this helps!'
        label, conf, rat, spans, lim = _parse_assessment_response(raw, "c8", ["ev8"])
        self.assertEqual(label, AssessmentLabel.SUPPORTED)
        self.assertAlmostEqual(conf, 0.9)
        self.assertEqual(rat, "Clear match.")



class TestLlamaAssessorWithMockedClient(unittest.TestCase):
    """Test LlamaAssessor with a mock OpenAI client (no server required)."""

    def _make_mock_client(self, response_content: str):
        """Create a mock OpenAI client that returns the given content."""
        mock_client = MagicMock()

        # Mock the models.list() call
        mock_model = MagicMock()
        mock_model.id = "test-model-3b"
        mock_client.models.list.return_value = MagicMock(data=[mock_model])

        # Mock the chat completion response
        mock_message = MagicMock()
        mock_message.content = response_content

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 150
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 200

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_successful_assessment_with_mock_client(self):
        response_json = '{"label": "SUPPORTED", "confidence": 0.8, "rationale": "Evidence confirms the claim.", "supporting_spans": ["Published in Philadelphia in 1844."], "limitations": null}'
        mock_client = self._make_mock_client(response_json)

        assessor = LlamaAssessor(client=mock_client, model_name="test-model-3b")
        claim = _make_claim("c_llama_1", "Arthur's Magazine was published in Philadelphia.")
        evidence = [
            _make_passage("ev_llama_1", "Arthur's Magazine",
                         "Arthur's Magazine was published in Philadelphia in 1844.")
        ]

        result = assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.SUPPORTED)
        self.assertAlmostEqual(result.confidence, 0.8)
        self.assertEqual(result.claim_id, "c_llama_1")
        self.assertEqual(result.cost.model_calls, 1)
        self.assertEqual(result.cost.prompt_tokens, 150)
        self.assertEqual(result.cost.completion_tokens, 50)
        self.assertEqual(result.cost.total_tokens, 200)
        self.assertIsNotNone(result.cost.latency_seconds)

    def test_empty_evidence_skips_llm_call(self):
        mock_client = self._make_mock_client("")
        assessor = LlamaAssessor(client=mock_client, model_name="test-model-3b")
        claim = _make_claim("c_llama_2", "Test claim.")

        result = assessor.assess(claim, [])
        self.assertEqual(result.label, AssessmentLabel.INSUFFICIENT)
        self.assertEqual(result.cost.model_calls, 0)
        # Verify the LLM was NOT called
        mock_client.chat.completions.create.assert_not_called()

    def test_llm_error_returns_insufficient(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("Server unavailable")

        assessor = LlamaAssessor(client=mock_client, model_name="test-model-3b")
        claim = _make_claim("c_llama_3", "Test claim.")
        evidence = [_make_passage("ev_llama_3", "Doc", "Some text.")]

        result = assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.INSUFFICIENT)
        self.assertLess(result.confidence, 0.5)
        self.assertIn("failed", result.rationale.lower())
        self.assertEqual(result.cost.model_calls, 1)

    def test_malformed_llm_response_returns_insufficient(self):
        mock_client = self._make_mock_client("Sorry, I can't help with that.")
        assessor = LlamaAssessor(client=mock_client, model_name="test-model-3b")
        claim = _make_claim("c_llama_4", "Test claim.")
        evidence = [_make_passage("ev_llama_4", "Doc", "Some relevant evidence text.")]

        result = assessor.assess(claim, evidence)
        self.assertEqual(result.label, AssessmentLabel.INSUFFICIENT)
        self.assertLess(result.confidence, 0.5)


class TestEvidenceModelEdgeCases(unittest.TestCase):
    """Test edge cases in evidence data models."""

    def test_verification_cost_serialization(self):
        cost = VerificationCost(
            model_calls=2,
            prompt_tokens=300,
            completion_tokens=100,
            total_tokens=400,
            latency_seconds=1.2345,
            model_id="llama-3.2-3b",
        )
        d = cost.to_dict()
        self.assertEqual(d["model_calls"], 2)
        self.assertEqual(d["latency_seconds"], 1.2345)
        self.assertEqual(d["model_id"], "llama-3.2-3b")

    def test_verification_cost_with_none_fields(self):
        cost = VerificationCost(model_calls=0)
        d = cost.to_dict()
        self.assertIsNone(d["prompt_tokens"])
        self.assertIsNone(d["latency_seconds"])
        self.assertIsNone(d["model_id"])

    def test_evidence_passage_with_metadata(self):
        passage = EvidencePassage(
            evidence_id="ep1",
            doc_title="Test Doc",
            source_text="Some text.",
            retrieval_score=0.88,
            metadata={"corpus": "hotpotqa", "sent_ids": [0, 1]},
        )
        self.assertEqual(passage.metadata["corpus"], "hotpotqa")

    def test_assessment_label_enum_values(self):
        self.assertEqual(AssessmentLabel.SUPPORTED.value, "SUPPORTED")
        self.assertEqual(AssessmentLabel.CONTRADICTED.value, "CONTRADICTED")
        self.assertEqual(AssessmentLabel.INSUFFICIENT.value, "INSUFFICIENT")
        self.assertEqual(AssessmentLabel.CONFLICTED.value, "CONFLICTED")


class TestLlamaAssessorBudgetAndErrors(unittest.TestCase):
    """Regression tests for Stage D.5 context budgeting and error classification."""

    def test_passage_budgeting_omits_lowest_ranked(self):
        """Passages exceeding max_passages (4) must be omitted and noted in limitations."""
        from src.evidence.llama_assessor import _format_evidence_block

        passages = [
            _make_passage(f"p{i}", f"Doc {i}", f"Text for doc {i}.", score=float(i))
            for i in range(6)
        ]
        # p5 has highest score (5.0), p0 has lowest (0.0)
        block, was_truncated, note = _format_evidence_block(passages, max_passages=4)
        self.assertTrue(was_truncated)
        self.assertIn("Omitted 2 lower-ranked passages", note)
        self.assertIn("p5", block)
        self.assertIn("p4", block)
        self.assertIn("p3", block)
        self.assertIn("p2", block)
        self.assertNotIn("p0", block)
        self.assertNotIn("p1", block)

    def test_passage_text_truncation(self):
        """Individual passage exceeding max_passage_chars must be truncated safely."""
        from src.evidence.llama_assessor import _format_evidence_block

        long_text = "A" * 1200
        passages = [_make_passage("p_long", "Doc Long", long_text, score=1.0)]
        block, was_truncated, note = _format_evidence_block(passages, max_passage_chars=900)
        self.assertTrue(was_truncated)
        self.assertIn("Truncated 1 passage(s) to 900 chars", note)
        self.assertIn("... [TRUNCATED]", block)
        self.assertLess(len(block), 1100)

    def test_technical_error_flags_is_error(self):
        """Exceptions in LLM completion must flag is_error=True and error_type."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Context window exceeded: 2302 > 2048")

        assessor = LlamaAssessor(client=mock_client, model_name="test-model")
        claim = _make_claim("c_err", "Some claim")
        evidence = [_make_passage("e1", "Doc", "Evidence text")]

        result = assessor.assess(claim, evidence)
        self.assertTrue(result.is_error)
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("Context window exceeded", result.rationale)

    def test_parse_error_flags_is_error(self):
        """Malformed JSON response must flag is_error=True and error_type='JSONDecodeError'."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "test-model"
        mock_client.models.list.return_value = MagicMock(data=[mock_model])

        mock_msg = MagicMock()
        mock_msg.content = "I cannot fulfill this request."
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock(choices=[mock_choice], usage=None)
        mock_client.chat.completions.create.return_value = mock_response

        assessor = LlamaAssessor(client=mock_client, model_name="test-model")
        claim = _make_claim("c_parse_err", "Some claim")
        evidence = [_make_passage("e1", "Doc", "Evidence text")]

        result = assessor.assess(claim, evidence)
        self.assertTrue(result.is_error)
        self.assertEqual(result.error_type, "JSONDecodeError")
        self.assertIn("Failed to parse", result.rationale)


if __name__ == "__main__":
    unittest.main()
