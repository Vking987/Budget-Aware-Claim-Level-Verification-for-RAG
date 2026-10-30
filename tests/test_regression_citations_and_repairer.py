"""Regression tests for citations, grounding integrity, and repairer selection.

Verifies:
1. Live configuration selects LLMRepairer; dry-run selects MockRepairer.
2. Failed live LLM repair does not silently fall back to MockRepairer or fabricate text.
3. Placeholder citation titles ('[Document Title]', '[Title]', '[Passage 1]') are never emitted.
4. Passages from non-verified claims are not leaked into citations.
5. Grounding requires verified claims with valid citations; not True merely because a claim was marked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus
from src.controller.interfaces import LLMRepairer, MockRepairer
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost
from src.pipeline import RAGVerificationPipeline
from src.synthesis.synthesizer import LLMAnswerSynthesizer, MockAnswerSynthesizer


def _create_claim(claim_id: str, text: str, state: ClaimState = ClaimState.PENDING) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        source_sentence_id="s0",
        original_text=text,
        normalized_text=text,
        claim_type=ClaimType.FACTUAL_RELATION,
        decomposition_status=DecompositionStatus.ATOMIC,
        state=state,
    )


class TestRepairerSelection:
    def test_live_configuration_selects_llm_repairer(self):
        """Live pipeline (dry_run=False) must configure LLMRepairer, not MockRepairer."""
        pipeline = RAGVerificationPipeline(dry_run=False)
        assert isinstance(pipeline.repairer, LLMRepairer)
        assert not isinstance(pipeline.repairer, MockRepairer)

    def test_dry_run_configuration_selects_mock_repairer(self):
        """Dry-run pipeline (dry_run=True) must configure MockRepairer."""
        pipeline = RAGVerificationPipeline(dry_run=True)
        assert isinstance(pipeline.repairer, MockRepairer)

    def test_failed_live_repair_raises_runtime_error_not_mock(self):
        """Failed LLMRepairer call raises RuntimeError and never invokes MockRepairer."""
        client = MagicMock()
        client.chat.completions.create.side_effect = ConnectionError("llama-server unavailable")
        repairer = LLMRepairer(client=client, model_name="test_model")

        claim = _create_claim("c1", "The Earth is flat.")
        assessment = EvidenceAssessment(
            claim_id="c1",
            evidence_ids=["e1"],
            supporting_spans={},
            label=AssessmentLabel.CONTRADICTED,
            confidence=0.95,
            rationale="Contradicted by planetary evidence.",
        )
        passages = [
            EvidencePassage(evidence_id="e1", doc_title="Earth Science", source_text="Earth is an oblate spheroid.")
        ]

        with pytest.raises(RuntimeError, match="LLMRepairer call failed"):
            repairer.repair(claim, assessment, passages)


class TestCitationIntegrity:
    def test_no_placeholder_citation_titles_in_mock_synthesizer(self):
        """Passages with None, empty, or 'Passage N' titles must never appear in cited_sources."""
        synthesizer = MockAnswerSynthesizer()
        claim = _create_claim("c1", "Water boils at 100 degrees Celsius.", ClaimState.VERIFIED)

        # Passages with None and placeholder titles
        passages = [
            EvidencePassage(evidence_id="e1", doc_title="", source_text="Water boils at 100C."),
            EvidencePassage(evidence_id="e2", doc_title="Passage 1", source_text="Boiling point is 100C."),
            EvidencePassage(evidence_id="e3", doc_title="Document Title", source_text="Boiling point physics."),
            EvidencePassage(evidence_id="e4", doc_title="Thermodynamics Manual", source_text="Measured at 100C at 1 atm."),
        ]

        result = synthesizer.synthesize(
            question="What temperature does water boil at?",
            initial_draft_answer="Water boils at 100C.",
            claims=[claim],
            evidence_store={"c1": passages},
        )

        assert "Thermodynamics Manual" in result.cited_sources
        assert "Document Title" not in result.cited_sources
        assert "Passage 1" not in result.cited_sources
        assert "[Document Title]" not in result.final_answer
        assert "[Passage 1]" not in result.final_answer

    def test_irrelevant_passage_leakage_prevented(self):
        """Passages associated with non-verified / unverified claims must not be cited."""
        synthesizer = MockAnswerSynthesizer()
        c_verified = _create_claim("c1", "Paris is the capital of France.", ClaimState.VERIFIED)
        c_unresolved = _create_claim("c2", "Mars has oceans of liquid water.", ClaimState.UNRESOLVED)

        evidence_store = {
            "c1": [EvidencePassage(evidence_id="e1", doc_title="French Geography", source_text="Paris is France capital.")],
            "c2": [EvidencePassage(evidence_id="e2", doc_title="Martian Astronomy", source_text="Mars has frozen ice poles.")],
        }

        result = synthesizer.synthesize(
            question="Tell me about Paris and Mars.",
            initial_draft_answer="Paris is capital of France and Mars has water.",
            claims=[c_verified, c_unresolved],
            evidence_store=evidence_store,
        )

        assert "French Geography" in result.cited_sources
        assert "Martian Astronomy" not in result.cited_sources

    def test_grounding_status_requires_resolved_evidence(self):
        """is_grounded must be False if verified claims have zero valid cited sources."""
        synthesizer = MockAnswerSynthesizer()
        claim = _create_claim("c1", "Some verified statement.", ClaimState.VERIFIED)

        # Evidence store has no valid titles (only empty / placeholders)
        evidence_store = {
            "c1": [EvidencePassage(evidence_id="e1", doc_title="", source_text="Some text.")],
        }

        result = synthesizer.synthesize(
            question="Question?",
            initial_draft_answer="Some statement.",
            claims=[claim],
            evidence_store=evidence_store,
        )

        assert result.cited_sources == []
        assert result.is_grounded is False
