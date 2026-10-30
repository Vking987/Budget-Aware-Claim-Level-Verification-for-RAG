"""Data models for evidence alignment.

Defines structured representations for evidence passages, assessment outcomes,
and verification cost tracking.

CRITICAL DESIGN NOTE:
    An EvidenceAssessment with label SUPPORTED is an assessor's judgment about
    claim-evidence alignment. It is NOT equivalent to ClaimState.VERIFIED.
    The controller (Stage B) is responsible for deciding whether an assessment
    warrants a lifecycle state transition through the validated transition rules
    in src/claims/models.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AssessmentLabel(str, Enum):
    """Assessment outcome labels for claim-evidence alignment.

    These labels describe the relationship between a claim and retrieved evidence,
    NOT the final verification decision.

    - SUPPORTED: The evidence directly and specifically entails the claim.
    - CONTRADICTED: The evidence directly and specifically contradicts the claim.
    - INSUFFICIENT: The evidence is absent, irrelevant, or does not address
                    the claim with enough specificity to decide.
    - CONFLICTED: Multiple evidence passages support contradictory conclusions
                  about the same claim.
    """
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"
    CONFLICTED = "CONFLICTED"


@dataclass
class EvidencePassage:
    """A single retrieved evidence passage with provenance metadata.

    Attributes:
        evidence_id: Unique identifier for this passage (e.g., Qdrant point ID
                     or a synthetic ID for testing).
        doc_title: Document title from the corpus (matches Qdrant payload["title"]).
        source_text: Full text content of the passage.
        retrieval_score: Cosine similarity or other retrieval score. None if
                        the passage was not obtained via retrieval (e.g., manually
                        provided ground truth).
        metadata: Optional additional metadata (e.g., corpus source, sentence indices).
    """
    evidence_id: str
    doc_title: str
    source_text: str
    retrieval_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VerificationCost:
    """Tracks the actual computational cost of a single assessment.

    All fields are optional because not all verifiers can measure all costs.
    Fields that cannot be measured are left as None rather than estimated.

    Attributes:
        model_calls: Number of LLM inference calls made.
        prompt_tokens: Prompt token count (if reported by the model API).
        completion_tokens: Completion token count (if reported by the model API).
        total_tokens: Total token count (if reported by the model API).
        latency_seconds: Wall-clock time for the assessment.
        model_id: Identifier of the model used (e.g., "llama-3.2-3b-instruct").
    """
    model_calls: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_seconds: Optional[float] = None
    model_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_seconds": round(self.latency_seconds, 4) if self.latency_seconds is not None else None,
            "model_id": self.model_id,
        }


@dataclass
class EvidenceAssessment:
    """Structured output of claim-evidence alignment assessment.

    IMPORTANT: This is the assessor's judgment, not a final verification decision.
    The 'confidence' field is a model estimate of assessment reliability, NOT a
    calibrated probability. It should be used for ranking and flagging low-confidence
    assessments for human review, not as a posterior probability.

    Attributes:
        claim_id: ID of the claim being assessed (references ClaimRecord.claim_id).
        evidence_ids: IDs of the evidence passages considered.
        supporting_spans: Exact text spans from evidence that support or contradict
                         the claim, where identifiable. Each entry maps an evidence_id
                         to a list of relevant text excerpts.
        label: Assessment outcome (SUPPORTED, CONTRADICTED, INSUFFICIENT, CONFLICTED).
        confidence: Model's self-reported confidence in [0.0, 1.0]. This is a model
                   estimate, NOT a calibrated probability. See module docstring.
        rationale: Concise explanation of why this label was assigned.
        limitations: Any ambiguity, missing evidence, or caveats the assessor identified.
        cost: Verification cost for this assessment.
        timestamp: When the assessment was produced.
        assessor_id: Identifier of the assessor implementation that produced this result.
    """
    claim_id: str
    evidence_ids: List[str]
    supporting_spans: Dict[str, List[str]]
    label: AssessmentLabel
    confidence: float
    rationale: str
    limitations: Optional[str] = None
    cost: VerificationCost = field(default_factory=VerificationCost)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    assessor_id: str = "unknown"
    is_error: bool = False
    error_type: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary for JSON output and provenance logging."""
        return {
            "claim_id": self.claim_id,
            "evidence_ids": self.evidence_ids,
            "supporting_spans": self.supporting_spans,
            "label": self.label.value,
            "confidence": self.confidence,
            "confidence_note": "Model estimate, NOT calibrated probability.",
            "rationale": self.rationale,
            "limitations": self.limitations,
            "cost": self.cost.to_dict(),
            "timestamp": self.timestamp,
            "assessor_id": self.assessor_id,
            "is_error": self.is_error,
            "error_type": self.error_type,
        }

