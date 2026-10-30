"""Data models for post-verification final answer synthesis.

Captures structured synthesis outputs, claim-to-evidence attribution,
uncertainty notes, citations, and resource accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.evidence.models import VerificationCost


@dataclass
class SynthesisResult:
    """Structured output of the final answer synthesis stage."""

    question: str
    initial_draft_answer: str
    final_answer: str
    verified_claims: List[str] = field(default_factory=list)
    unresolved_claims: List[str] = field(default_factory=list)
    contradicted_claims: List[str] = field(default_factory=list)
    cited_sources: List[str] = field(default_factory=list)
    uncertainty_notes: List[str] = field(default_factory=list)
    is_grounded: bool = True
    has_unresolved_limitations: bool = False
    cost: VerificationCost = field(default_factory=lambda: VerificationCost(model_calls=0))
    synthesis_method: str = "llm_grounded"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None
    # Claim-faithful grounding (set by the LLM synthesizer): per-claim citation mapping, enforced-validation outcome
    claim_citations: List[Dict[str, Any]] = field(default_factory=list)
    answer_violations: List[str] = field(default_factory=list)
    polish_status: str = ""  # llm_accepted | llm_rejected_stitched | stitched_fallback | validation_failed_abstention
    grounding_enforced: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "question": self.question,
            "initial_draft_answer": self.initial_draft_answer,
            "final_answer": self.final_answer,
            "verified_claims": list(self.verified_claims),
            "unresolved_claims": list(self.unresolved_claims),
            "contradicted_claims": list(self.contradicted_claims),
            "cited_sources": list(self.cited_sources),
            "uncertainty_notes": list(self.uncertainty_notes),
            "is_grounded": self.is_grounded,
            "has_unresolved_limitations": self.has_unresolved_limitations,
            "cost": self.cost.to_dict() if self.cost else None,
            "synthesis_method": self.synthesis_method,
            "timestamp": self.timestamp,
            "error": self.error,
            "claim_citations": list(self.claim_citations),
            "answer_violations": list(self.answer_violations),
            "polish_status": self.polish_status,
            "grounding_enforced": self.grounding_enforced,
        }
