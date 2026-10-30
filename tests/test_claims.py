"""Unit tests for content-bearing filtering, claim decomposition, CTS, and state tracking."""

import unittest
from src.claims.models import (
    ClaimRecord,
    ClaimState,
    ClaimType,
    DecompositionStatus,
    InvalidStateTransitionError,
    SentenceRecord,
)
from src.claims.filter import ContentBearingFilter
from src.claims.decomposer import RuleBasedDecomposer
from src.claims.triage import ClaimTriageScorer, CTSConfig


class TestContentBearingFilter(unittest.TestCase):
    """Test suite for ContentBearingFilter."""

    def setUp(self) -> None:
        self.filter = ContentBearingFilter()

    def test_short_concise_answer(self):
        records = self.filter.process("Delhi.")
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].is_content_bearing)
        self.assertEqual(records[0].cleaned_text, "Delhi.")

    def test_verbose_answer_strips_boilerplate_preserves_content(self):
        text = "Based on the provided context, the Oberoi family is part of a hotel company that has a head office in Delhi."
        records = self.filter.process(text)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].is_content_bearing)
        self.assertEqual(
            records[0].cleaned_text,
            "The Oberoi family is part of a hotel company that has a head office in Delhi."
        )
        self.assertEqual(records[0].original_text, text)
        self.assertEqual(records[0].char_start, 0)
        self.assertEqual(records[0].char_end, len(text))

    def test_pure_conversational_filler_discarded(self):
        text = "Sure! Here is the answer:\nDelhi."
        records = self.filter.process(text)
        # Part 0: "Sure!" -> non-content
        # Part 1: "Here is the answer:" -> non-content
        # Part 2: "Delhi." -> content-bearing
        self.assertEqual(len(records), 3)
        self.assertFalse(records[0].is_content_bearing)
        self.assertFalse(records[1].is_content_bearing)
        self.assertTrue(records[2].is_content_bearing)
        self.assertEqual(records[2].cleaned_text, "Delhi.")

    def test_uncertainty_statement_strictly_preserved(self):
        text = "The context does not contain enough information to answer the question."
        records = self.filter.process(text)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].is_content_bearing)
        self.assertIn("safeguard_preserved", records[0].filter_reason)

    def test_negation_statement_strictly_preserved(self):
        text = "Neither Arthur's Magazine nor First for Women was started first."
        records = self.filter.process(text)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].is_content_bearing)
        self.assertIn("safeguard_preserved", records[0].filter_reason)

    def test_dates_and_numbers_strictly_preserved(self):
        text = "First for Women was started in 1989 and had a circulation of 1,310,696 copies."
        records = self.filter.process(text)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].is_content_bearing)
        self.assertIn("1989", records[0].cleaned_text)
        self.assertIn("1,310,696", records[0].cleaned_text)

    def test_multiple_sentences_with_offsets(self):
        text = "First for Women was started in 1989. It is based in Englewood Cliffs."
        records = self.filter.process(text)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].cleaned_text, "First for Women was started in 1989.")
        self.assertEqual(records[1].cleaned_text, "It is based in Englewood Cliffs.")
        self.assertEqual(records[0].char_start, 0)
        self.assertEqual(records[0].char_end, 36)
        self.assertEqual(records[1].char_start, 37)


class TestClaimDecomposer(unittest.TestCase):
    """Test suite for RuleBasedDecomposer."""

    def setUp(self) -> None:
        self.decomposer = RuleBasedDecomposer()

    def test_compound_verbs_with_shared_subject_split_faithfully(self):
        s = SentenceRecord(
            sentence_id="s_0",
            original_text="Arthur's Magazine was published in Philadelphia and was edited by T.S. Arthur.",
            cleaned_text="Arthur's Magazine was published in Philadelphia and was edited by T.S. Arthur.",
            is_content_bearing=True,
            filter_reason="declarative",
            char_start=0,
            char_end=78
        )
        claims = self.decomposer.decompose(s)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0].claim_id, "s_0_c0")
        self.assertEqual(claims[1].claim_id, "s_0_c1")
        self.assertEqual(claims[0].original_text, "Arthur's Magazine was published in Philadelphia.")
        self.assertEqual(claims[1].original_text, "Arthur's Magazine was edited by T.S. Arthur.")
        self.assertEqual(claims[0].decomposition_status, DecompositionStatus.SPLIT_DETERMINISTIC)
        self.assertEqual(claims[1].decomposition_status, DecompositionStatus.SPLIT_DETERMINISTIC)

    def test_appositive_construction_split(self):
        s = SentenceRecord(
            sentence_id="s_1",
            original_text="Peggy Seeger, an American folk singer, married James Miller.",
            cleaned_text="Peggy Seeger, an American folk singer, married James Miller.",
            is_content_bearing=True,
            filter_reason="declarative",
            char_start=0,
            char_end=60
        )
        claims = self.decomposer.decompose(s)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0].original_text, "Peggy Seeger married James Miller.")
        self.assertEqual(claims[1].original_text, "Peggy Seeger was an American folk singer.")

    def test_temporal_comparison_classification_and_preservation(self):
        s = SentenceRecord(
            sentence_id="s_2",
            original_text="Arthur's Magazine was published in 1844 earlier than First for Women in 1989.",
            cleaned_text="Arthur's Magazine was published in 1844 earlier than First for Women in 1989.",
            is_content_bearing=True,
            filter_reason="declarative",
            char_start=0,
            char_end=77
        )
        claims = self.decomposer.decompose(s)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].claim_type, ClaimType.TEMPORAL_COMPARISON)
        self.assertEqual(claims[0].decomposition_status, DecompositionStatus.UNSPLIT_PRESERVED)

    def test_negation_must_not_be_split(self):
        s = SentenceRecord(
            sentence_id="s_3",
            original_text="Neither Arthur's Magazine nor First for Women was started first.",
            cleaned_text="Neither Arthur's Magazine nor First for Women was started first.",
            is_content_bearing=True,
            filter_reason="negation",
            char_start=0,
            char_end=64
        )
        claims = self.decomposer.decompose(s)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].claim_type, ClaimType.EXISTENCE_NEGATION)
        self.assertEqual(claims[0].decomposition_status, DecompositionStatus.UNSPLIT_PRESERVED)

    def test_collective_predicate_must_not_be_split(self):
        s = SentenceRecord(
            sentence_id="s_4",
            original_text="Arthur's Magazine and First for Women were both published in the United States.",
            cleaned_text="Arthur's Magazine and First for Women were both published in the United States.",
            is_content_bearing=True,
            filter_reason="declarative",
            char_start=0,
            char_end=79
        )
        claims = self.decomposer.decompose(s)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].decomposition_status, DecompositionStatus.UNSPLIT_PRESERVED)

    def test_causal_statement_preserved_intact(self):
        s = SentenceRecord(
            sentence_id="s_5",
            original_text="Because the magazine was unprofitable, it was merged into Godey's Lady's Book in 1846.",
            cleaned_text="Because the magazine was unprofitable, it was merged into Godey's Lady's Book in 1846.",
            is_content_bearing=True,
            filter_reason="declarative",
            char_start=0,
            char_end=86
        )
        claims = self.decomposer.decompose(s)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].decomposition_status, DecompositionStatus.UNSPLIT_PRESERVED)


class TestClaimTriageScorer(unittest.TestCase):
    """Test suite for ClaimTriageScorer (CTS)."""

    def setUp(self) -> None:
        self.scorer = ClaimTriageScorer()

    def test_cts_feature_bounds_and_determinism(self):
        claim = ClaimRecord(
            claim_id="c_0",
            source_sentence_id="s_0",
            original_text="First for Women was started in 1989 in Englewood Cliffs.",
            normalized_text="First for Women was started in 1989 in Englewood Cliffs.",
            claim_type=ClaimType.TEMPORAL_COMPARISON,
            decomposition_status=DecompositionStatus.ATOMIC
        )
        context = "First for Women is a magazine published by Bauer Media Group in the USA."

        # Bounded between 0 and 1
        score1 = self.scorer.score_claim(claim, context)
        self.assertTrue(0.0 <= score1 <= 1.0)
        self.assertIsNotNone(claim.triage_features)

        # Check feature bounds
        for fname, fval in claim.triage_features.items():
            self.assertTrue(0.0 <= fval <= 1.0, f"Feature {fname} = {fval} out of bounds")

        # Determinism check
        score2 = self.scorer.score_claim(claim, context)
        self.assertEqual(score1, score2)

    def test_missing_or_empty_context_handling(self):
        claim = ClaimRecord(
            claim_id="c_1",
            source_sentence_id="s_1",
            original_text="Cadmium chloride has a molecular weight of 183.32.",
            normalized_text="Cadmium chloride has a molecular weight of 183.32.",
            claim_type=ClaimType.NUMERICAL_QUANTITATIVE,
            decomposition_status=DecompositionStatus.ATOMIC
        )
        score_none = self.scorer.score_claim(claim, None)
        self.assertEqual(claim.triage_features["novelty"], 1.0)
        self.assertTrue(0.0 <= score_none <= 1.0)

        score_empty = self.scorer.score_claim(claim, "   ")
        self.assertEqual(score_none, score_empty)

    def test_score_ordering_under_controlled_conditions(self):
        # Claim A: Specific temporal comparison completely absent from context (high novelty, high specificity)
        claim_a = ClaimRecord(
            claim_id="c_a",
            source_sentence_id="s_a",
            original_text="Arthur's Magazine was founded in 1844 earlier than First for Women in 1989.",
            normalized_text="Arthur's Magazine was founded in 1844 earlier than First for Women in 1989.",
            claim_type=ClaimType.TEMPORAL_COMPARISON,
            decomposition_status=DecompositionStatus.ATOMIC
        )

        # Claim B: Vague general claim fully grounded in context (low novelty, low specificity)
        claim_b = ClaimRecord(
            claim_id="c_b",
            source_sentence_id="s_b",
            original_text="The magazine was published.",
            normalized_text="The magazine was published.",
            claim_type=ClaimType.UNKNOWN_GENERAL,
            decomposition_status=DecompositionStatus.ATOMIC
        )

        context = "The magazine was published in the United States."

        score_a = self.scorer.score_claim(claim_a, context)
        score_b = self.scorer.score_claim(claim_b, context)

        # High priority claim A must strictly outscore low priority claim B
        self.assertGreater(score_a, score_b)

    def test_cts_config_validation(self):
        # Weights must sum to 1.0
        with self.assertRaises(ValueError):
            CTSConfig(weight_claim_type=0.5, weight_specificity=0.5, weight_novelty=0.5, weight_complexity=0.5)


class TestClaimStateTransitions(unittest.TestCase):
    """Test suite for explicit claim lifecycle state machine."""

    def test_initial_state_is_pending(self):
        claim = ClaimRecord(
            claim_id="c_init",
            source_sentence_id="s_init",
            original_text="Sample text.",
            normalized_text="Sample text.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC
        )
        self.assertEqual(claim.state, ClaimState.PENDING)
        self.assertEqual(len(claim.state_history), 1)

    def test_valid_lifecycle_transitions(self):
        claim = ClaimRecord(
            claim_id="c_valid",
            source_sentence_id="s_valid",
            original_text="Arthur's Magazine was published in Philadelphia.",
            normalized_text="Arthur's Magazine was published in Philadelphia.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC
        )

        # PENDING -> SELECTED
        claim.transition_to(ClaimState.SELECTED, reason="Selected by controller")
        self.assertEqual(claim.state, ClaimState.SELECTED)

        # SELECTED -> VERIFIED (with evidence)
        evidence = {"doc_title": "Arthur's Magazine", "match_score": 0.95}
        claim.transition_to(ClaimState.VERIFIED, reason="Confirmed in document", evidence=evidence)
        self.assertEqual(claim.state, ClaimState.VERIFIED)
        self.assertEqual(claim.evidence_provenance, evidence)
        self.assertEqual(len(claim.state_history), 3)

    def test_deferred_claims_remain_unverified(self):
        claim = ClaimRecord(
            claim_id="c_def",
            source_sentence_id="s_def",
            original_text="Minor secondary fact.",
            normalized_text="Minor secondary fact.",
            claim_type=ClaimType.UNKNOWN_GENERAL,
            decomposition_status=DecompositionStatus.ATOMIC
        )

        # PENDING -> DEFERRED
        claim.transition_to(ClaimState.DEFERRED, reason="Budget exhausted")
        self.assertEqual(claim.state, ClaimState.DEFERRED)
        self.assertNotEqual(claim.state, ClaimState.VERIFIED)

    def test_illegal_jump_to_verified_prevented(self):
        claim = ClaimRecord(
            claim_id="c_illegal",
            source_sentence_id="s_illegal",
            original_text="Some assertion.",
            normalized_text="Some assertion.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC
        )

        # Cannot jump PENDING -> VERIFIED
        with self.assertRaises(InvalidStateTransitionError):
            claim.transition_to(ClaimState.VERIFIED, reason="Illegal jump", evidence={"doc": "test"})

    def test_verified_without_evidence_rejected(self):
        claim = ClaimRecord(
            claim_id="c_no_ev",
            source_sentence_id="s_no_ev",
            original_text="Some assertion.",
            normalized_text="Some assertion.",
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC
        )
        claim.transition_to(ClaimState.SELECTED, reason="Selected")

        # SELECTED -> VERIFIED without evidence must fail
        with self.assertRaises(InvalidStateTransitionError):
            claim.transition_to(ClaimState.VERIFIED, reason="Trying to verify without evidence", evidence=None)


if __name__ == "__main__":
    unittest.main()
