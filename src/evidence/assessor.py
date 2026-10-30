"""Abstract evidence assessor interface and deterministic mock implementation.

The assessor interface defines the contract for claim-evidence alignment:
given a claim and a set of evidence passages, produce a structured
EvidenceAssessment.

This module provides:
- BaseAssessor: Abstract interface that all assessor implementations must follow.
- MockDeterministicAssessor: A fully deterministic, rule-based assessor for
  offline unit testing. It uses keyword matching and structural heuristics
  to simulate assessment outcomes WITHOUT requiring an LLM.

DESIGN PRINCIPLES:
1. Assessors produce EvidenceAssessments, NOT ClaimState transitions.
2. A high retrieval score does not independently establish support.
3. When evidence is ambiguous, return INSUFFICIENT rather than forcing a verdict.
4. Negation, dates, numerical values, and temporal orderings must be preserved
   in the assessment logic.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from src.claims.models import ClaimRecord
from .models import (
    AssessmentLabel,
    EvidenceAssessment,
    EvidencePassage,
    VerificationCost,
)


class BaseAssessor(ABC):
    """Abstract interface for claim-evidence alignment assessors.

    All assessor implementations must:
    1. Accept a ClaimRecord and a list of EvidencePassages.
    2. Return a single EvidenceAssessment.
    3. Record verification costs (model calls, tokens, latency).
    4. Never produce SUPPORTED based on retrieval score alone.
    5. Return INSUFFICIENT when evidence is absent, irrelevant, or ambiguous.
    """

    @property
    @abstractmethod
    def assessor_id(self) -> str:
        """Unique identifier for this assessor implementation."""
        ...

    @abstractmethod
    def assess(
        self,
        claim: ClaimRecord,
        evidence: List[EvidencePassage],
    ) -> EvidenceAssessment:
        """Assess claim-evidence alignment.

        Args:
            claim: The atomic claim to verify.
            evidence: Zero or more retrieved evidence passages.

        Returns:
            A structured EvidenceAssessment with label, confidence,
            rationale, supporting spans, limitations, and cost.
        """
        ...


class MockDeterministicAssessor(BaseAssessor):
    """Deterministic, rule-based assessor for offline testing.

    This assessor uses keyword matching and structural heuristics to produce
    predictable, hand-verifiable assessment outcomes. It does NOT use an LLM.

    Rules (applied in order):
    1. No evidence → INSUFFICIENT.
    2. Evidence explicitly contradicts the claim (negation mismatch) → CONTRADICTED.
    3. Multiple passages with conflicting signals → CONFLICTED.
    4. Evidence contains sufficient keyword overlap with the claim → SUPPORTED.
    5. Otherwise → INSUFFICIENT.

    IMPORTANT: A high retrieval score ALONE never produces SUPPORTED.
    The mock assessor requires actual textual content overlap between the
    claim and evidence to label support.

    Limitations:
    - No semantic understanding; purely lexical.
    - Cannot detect paraphrases, implications, or indirect support.
    - Serves as a structural test fixture, not an accurate verifier.
    """

    @property
    def assessor_id(self) -> str:
        return "mock_deterministic_v1"

    def _extract_content_tokens(self, text: str) -> set:
        """Extract lowercase content words, stripping punctuation."""
        if not text:
            return set()
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "was", "are", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "it", "its", "this", "that", "they", "their", "them", "which",
            "who", "whom", "what", "when", "where", "why", "how",
        }
        return {w for w in clean.split() if len(w) > 1 and w not in stopwords}

    def _extract_numbers(self, text: str) -> set:
        """Extract all numeric tokens (integers and decimals)."""
        return set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text))

    def _extract_years(self, text: str) -> set:
        """Extract 4-digit years."""
        return set(re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", text))

    def _has_negation(self, text: str) -> bool:
        """Check if text contains explicit negation markers."""
        return bool(re.search(
            r"\b(?:not|no|neither|nor|never|none|cannot|didn't|doesn't|wasn't|weren't|isn't|aren't)\b",
            text, re.IGNORECASE
        ))

    def _check_temporal_contradiction(self, claim_text: str, evidence_text: str) -> bool:
        """Check for temporal ordering contradictions.

        Example: Claim says "founded in 1844" but evidence says "founded in 1920".
        """
        claim_years = self._extract_years(claim_text)
        evidence_years = self._extract_years(evidence_text)

        # If both contain years, but claim years are not supported by evidence years
        if claim_years and evidence_years:
            if not claim_years.issubset(evidence_years):
                claim_tokens = self._extract_content_tokens(claim_text)
                evidence_tokens = self._extract_content_tokens(evidence_text)
                shared_context = claim_tokens & evidence_tokens
                year_context = {
                    "founded", "started", "born", "died", "published",
                    "released", "established", "created", "opened"
                }
                if len(shared_context) >= 2 and (shared_context & year_context):
                    return True

        return False

    def _check_numerical_contradiction(self, claim_text: str, evidence_text: str) -> bool:
        """Check for numerical value contradictions.

        Example: Claim says "circulation of 1,310,696" but evidence says "circulation of 500,000".
        """
        claim_numbers = self._extract_numbers(claim_text)
        evidence_numbers = self._extract_numbers(evidence_text)

        if not claim_numbers or not evidence_numbers:
            return False

        # If all claim numbers appear in the evidence, no numerical contradiction
        if claim_numbers.issubset(evidence_numbers):
            return False

        # If both mention numbers but the specific numbers don't match,
        # and they share entity context, this may be a contradiction
        claim_tokens = self._extract_content_tokens(claim_text)
        evidence_tokens = self._extract_content_tokens(evidence_text)
        shared_context = claim_tokens & evidence_tokens

        # Need shared context AND different numbers to flag contradiction
        if len(shared_context) >= 2:
            # Only flag if the numbers are in the same semantic context
            # (e.g., both about "circulation" or "population")
            context_words = {
                "circulation", "population", "copies", "revenue",
                "height", "weight", "distance", "area", "founded",
                "started", "born", "died", "age", "years"
            }
            if shared_context & context_words:
                return True

        return False

    def _compute_support_score(
        self,
        claim: ClaimRecord,
        passage: EvidencePassage
    ) -> tuple:
        """Compute a content-based support score for a single passage.

        Returns (score, spans) where score is a float in [0, 1] and
        spans is a list of matching text excerpts.

        This is based on content token overlap, entity matching, and
        numerical agreement — NOT retrieval score.
        """
        claim_text = claim.normalized_text or claim.original_text
        evidence_text = passage.source_text

        claim_tokens = self._extract_content_tokens(claim_text)
        evidence_tokens = self._extract_content_tokens(evidence_text)

        if not claim_tokens:
            return 0.0, []

        # Content overlap ratio
        overlap = claim_tokens & evidence_tokens
        overlap_ratio = len(overlap) / len(claim_tokens)

        # Year agreement bonus
        claim_years = self._extract_years(claim_text)
        evidence_years = self._extract_years(evidence_text)
        year_bonus = 0.15 if claim_years and claim_years.issubset(evidence_years) else 0.0

        # Number agreement bonus
        claim_numbers = self._extract_numbers(claim_text)
        evidence_numbers = self._extract_numbers(evidence_text)
        number_bonus = 0.10 if claim_numbers and claim_numbers.issubset(evidence_numbers) else 0.0

        score = min(1.0, overlap_ratio + year_bonus + number_bonus)

        # Extract matching spans (sentences containing overlap tokens)
        spans = []
        if overlap:
            sentences = re.split(r"(?<=[.!?])\s+", evidence_text)
            for sent in sentences:
                sent_tokens = self._extract_content_tokens(sent)
                if sent_tokens & overlap:
                    spans.append(sent.strip())

        return score, spans

    def assess(
        self,
        claim: ClaimRecord,
        evidence: List[EvidencePassage],
    ) -> EvidenceAssessment:
        """Assess claim-evidence alignment using deterministic rules."""
        start_time = time.monotonic()
        claim_text = claim.normalized_text or claim.original_text

        # Rule 1: No evidence → INSUFFICIENT
        if not evidence:
            elapsed = time.monotonic() - start_time
            return EvidenceAssessment(
                claim_id=claim.claim_id,
                evidence_ids=[],
                supporting_spans={},
                label=AssessmentLabel.INSUFFICIENT,
                confidence=0.95,
                rationale="No evidence passages provided for assessment.",
                limitations="Cannot assess without evidence.",
                cost=VerificationCost(
                    model_calls=0,
                    latency_seconds=elapsed,
                ),
                assessor_id=self.assessor_id,
            )

        evidence_ids = [e.evidence_id for e in evidence]

        # Check each passage for support and contradiction signals
        support_signals = []
        contradiction_signals = []
        all_spans: dict = {}

        for passage in evidence:
            ev_text = passage.source_text
            if not ev_text or not ev_text.strip():
                continue

            # Check for direct contradictions
            claim_negated = self._has_negation(claim_text)
            evidence_negated = self._has_negation(ev_text)

            # Negation mismatch: one negates, the other doesn't, but they share entities
            claim_tokens = self._extract_content_tokens(claim_text)
            ev_tokens = self._extract_content_tokens(ev_text)
            shared = claim_tokens & ev_tokens

            is_contra = False
            if len(shared) >= 2 and claim_negated != evidence_negated:
                is_contra = True
                contradiction_signals.append(passage.evidence_id)
                # Extract contradicting spans
                sentences = re.split(r"(?<=[.!?])\s+", ev_text)
                contra_spans = [s.strip() for s in sentences if self._has_negation(s) != claim_negated]
                if contra_spans:
                    all_spans[passage.evidence_id] = contra_spans

            # Check temporal contradictions
            if not is_contra and self._check_temporal_contradiction(claim_text, ev_text):
                is_contra = True
                contradiction_signals.append(passage.evidence_id)

            # Check numerical contradictions
            if not is_contra and self._check_numerical_contradiction(claim_text, ev_text):
                is_contra = True
                contradiction_signals.append(passage.evidence_id)

            if is_contra:
                if passage.evidence_id not in all_spans:
                    sentences = re.split(r"(?<=[.!?])\s+", ev_text)
                    matched_sents = [s.strip() for s in sentences if self._extract_content_tokens(s) & shared]
                    if matched_sents:
                        all_spans[passage.evidence_id] = matched_sents
                # A contradicting passage does not contribute support
                continue

            # Check support
            score, spans = self._compute_support_score(claim, passage)
            if score >= 0.5:
                support_signals.append((passage.evidence_id, score))
                if spans:
                    all_spans[passage.evidence_id] = spans

        elapsed = time.monotonic() - start_time
        cost = VerificationCost(model_calls=0, latency_seconds=elapsed)

        # Rule 3: Conflicting signals → CONFLICTED
        if support_signals and contradiction_signals:
            return EvidenceAssessment(
                claim_id=claim.claim_id,
                evidence_ids=evidence_ids,
                supporting_spans=all_spans,
                label=AssessmentLabel.CONFLICTED,
                confidence=0.5,
                rationale=(
                    f"Evidence contains both supporting and contradicting signals. "
                    f"Supporting passages: {[s[0] for s in support_signals]}. "
                    f"Contradicting passages: {contradiction_signals}."
                ),
                limitations=(
                    "Lexical conflict detection cannot resolve semantic nuance. "
                    "Manual review recommended."
                ),
                cost=cost,
                assessor_id=self.assessor_id,
            )

        # Rule 2: Direct contradiction → CONTRADICTED
        if contradiction_signals and not support_signals:
            return EvidenceAssessment(
                claim_id=claim.claim_id,
                evidence_ids=evidence_ids,
                supporting_spans=all_spans,
                label=AssessmentLabel.CONTRADICTED,
                confidence=0.7,
                rationale=(
                    f"Evidence contradicts the claim via negation mismatch, temporal ordering, "
                    f"or numerical disagreement. Contradicting passages: {contradiction_signals}."
                ),
                limitations="Lexical contradiction detection; may miss paraphrased contradictions.",
                cost=cost,
                assessor_id=self.assessor_id,
            )

        # Rule 4: Sufficient content support → SUPPORTED
        if support_signals:
            best_score = max(s[1] for s in support_signals)
            return EvidenceAssessment(
                claim_id=claim.claim_id,
                evidence_ids=evidence_ids,
                supporting_spans=all_spans,
                label=AssessmentLabel.SUPPORTED,
                confidence=min(0.85, best_score),
                rationale=(
                    f"Evidence content supports the claim with token overlap score "
                    f"{best_score:.2f}. Supporting passages: {[s[0] for s in support_signals]}."
                ),
                limitations="Lexical overlap support; cannot detect paraphrased or implicit support.",
                cost=cost,
                assessor_id=self.assessor_id,
            )

        # Rule 5: No sufficient signals → INSUFFICIENT
        return EvidenceAssessment(
            claim_id=claim.claim_id,
            evidence_ids=evidence_ids,
            supporting_spans=all_spans,
            label=AssessmentLabel.INSUFFICIENT,
            confidence=0.6,
            rationale=(
                "Evidence passages do not contain sufficient content overlap or "
                "semantic indicators to support or contradict the claim."
            ),
            limitations="Lexical matching only; semantically relevant but differently worded evidence will be missed.",
            cost=cost,
            assessor_id=self.assessor_id,
        )
