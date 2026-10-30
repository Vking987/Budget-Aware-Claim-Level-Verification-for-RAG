"""Evidence alignment package for RAG claim verification.

Exposes:
- Data models: EvidencePassage, EvidenceAssessment, AssessmentLabel, VerificationCost
- Assessor interface: BaseAssessor
- Implementations: MockDeterministicAssessor, LlamaAssessor
"""

from .models import (
    AssessmentLabel,
    EvidenceAssessment,
    EvidencePassage,
    VerificationCost,
)
from .assessor import BaseAssessor, MockDeterministicAssessor
from .llama_assessor import LlamaAssessor

__all__ = [
    "AssessmentLabel",
    "EvidenceAssessment",
    "EvidencePassage",
    "VerificationCost",
    "BaseAssessor",
    "MockDeterministicAssessor",
    "LlamaAssessor",
]
