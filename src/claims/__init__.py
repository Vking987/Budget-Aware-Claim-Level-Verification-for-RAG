"""Claims processing package for RAG verification pipeline.

Exposes:
- Data models & state machine: ClaimRecord, SentenceRecord, ClaimState, ClaimType, DecompositionStatus
- Filtering: ContentBearingFilter
- Decomposition: BaseDecomposer, RuleBasedDecomposer
- Triage: ClaimTriageScorer, CTSConfig
"""

from .models import (
    ClaimRecord,
    ClaimState,
    ClaimType,
    DecompositionStatus,
    InvalidStateTransitionError,
    SentenceRecord,
    VALID_TRANSITIONS,
)
from .filter import ContentBearingFilter
from .decomposer import BaseDecomposer, RuleBasedDecomposer
from .triage import ClaimTriageScorer, CTSConfig

__all__ = [
    "ClaimRecord",
    "ClaimState",
    "ClaimType",
    "DecompositionStatus",
    "InvalidStateTransitionError",
    "SentenceRecord",
    "VALID_TRANSITIONS",
    "ContentBearingFilter",
    "BaseDecomposer",
    "RuleBasedDecomposer",
    "ClaimTriageScorer",
    "CTSConfig",
]
