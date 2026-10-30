"""Data models, enumerations, and state transition machine for claim processing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ClaimState(str, Enum):
    """Explicit lifecycle states for claim verification.

    - PENDING: Initial state upon extraction and decomposition.
    - SELECTED: Chosen by the controller for verification.
    - DEFERRED: Held unverified under budget constraints (tracked for deferred-claim error rate).
    - VERIFIED: Formally confirmed supported by retrieved evidence.
    - UNRESOLVED: Verification attempted, but evidence was inconclusive, contradictory, or timed out.
    """
    PENDING = "PENDING"
    SELECTED = "SELECTED"
    DEFERRED = "DEFERRED"
    VERIFIED = "VERIFIED"
    UNRESOLVED = "UNRESOLVED"


class ClaimType(str, Enum):
    """Categorization of factual claim propositions."""
    FACTUAL_RELATION = "FACTUAL_RELATION"      # e.g., "X is married to Y", "Company X is based in City Y"
    TEMPORAL_COMPARISON = "TEMPORAL_COMPARISON" # e.g., "A was started before B", "Started in 1989 after 1844"
    NUMERICAL_QUANTITATIVE = "NUMERICAL_QUANTITATIVE" # e.g., "Circulation was 1,310,696", "Won 5 titles"
    EXISTENCE_NEGATION = "EXISTENCE_NEGATION"  # e.g., "Neither X nor Y won", "No evidence was found"
    ATTRIBUTION = "ATTRIBUTION"                # e.g., "According to documents, X stated Y"
    LIMITATION_UNCERTAINTY = "LIMITATION_UNCERTAINTY" # e.g., "The question does not specify which country", "Cannot be determined"
    REFUSAL_UNANSWERABLE = "REFUSAL_UNANSWERABLE" # e.g., "I cannot answer this question"
    UNKNOWN_GENERAL = "UNKNOWN_GENERAL"        # General declarative claim


class DecompositionStatus(str, Enum):
    """Status indicating how a claim was derived from source text."""
    ATOMIC = "ATOMIC"                          # Already atomic, no split required
    SPLIT_DETERMINISTIC = "SPLIT_DETERMINISTIC" # Split into independent propositions with high syntactic confidence
    SPLIT_UNCERTAIN = "SPLIT_UNCERTAIN"        # Split performed, but syntactic ambiguity exists (flagged for review)
    UNSPLIT_PRESERVED = "UNSPLIT_PRESERVED"    # Kept compound because splitting risks breaking semantics (e.g. collective predicate)


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal claim state transition is attempted."""
    pass


# Valid state transitions mapping
VALID_TRANSITIONS: Dict[ClaimState, Set[ClaimState]] = {
    ClaimState.PENDING: {ClaimState.SELECTED, ClaimState.DEFERRED},
    ClaimState.SELECTED: {ClaimState.VERIFIED, ClaimState.UNRESOLVED, ClaimState.DEFERRED},
    ClaimState.DEFERRED: {ClaimState.SELECTED},
    ClaimState.UNRESOLVED: {ClaimState.SELECTED},
    ClaimState.VERIFIED: set(),  # Terminal state
}


@dataclass
class SentenceRecord:
    """Represents an extracted sentence from an answer with provenance."""
    sentence_id: str
    original_text: str
    cleaned_text: str
    is_content_bearing: bool
    filter_reason: str
    char_start: int
    char_end: int
    confidence: float = 1.0
    is_substantive: bool = True
    is_limitation: bool = False
    is_refusal: bool = False


@dataclass
class ClaimRecord:
    """Represents an atomic or propositional claim extracted from an answer."""
    claim_id: str
    source_sentence_id: str
    original_text: str
    normalized_text: str
    claim_type: ClaimType
    decomposition_status: DecompositionStatus
    state: ClaimState = ClaimState.PENDING
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    triage_score: Optional[float] = None
    triage_features: Optional[Dict[str, float]] = None
    uncertainty_flag: bool = False
    is_substantive: bool = True
    is_limitation: bool = False
    evidence_provenance: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.state_history:
            self.state_history.append({
                "from_state": None,
                "to_state": self.state.value,
                "timestamp": datetime.now().isoformat(),
                "reason": "Initial decomposition"
            })

    def transition_to(
        self,
        new_state: ClaimState,
        reason: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> None:
        """Execute a validated lifecycle state transition.

        Args:
            new_state: Target ClaimState.
            reason: Explicit justification for the transition.
            evidence: Optional supporting evidence payload (required for VERIFIED).

        Raises:
            InvalidStateTransitionError: If the transition violates state rules or attempts
                                         to assign VERIFIED without supporting evidence.
        """
        allowed = VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidStateTransitionError(
                f"Cannot transition claim '{self.claim_id}' from {self.state.value} to {new_state.value}. "
                f"Allowed transitions: {[s.value for s in allowed]}."
            )

        # Invariant: A claim can NEVER be verified on triage score or retrieval alone
        if new_state == ClaimState.VERIFIED and not evidence:
            raise InvalidStateTransitionError(
                f"Cannot transition claim '{self.claim_id}' to VERIFIED without verified evidence payload."
            )

        old_state = self.state
        self.state = new_state
        if evidence is not None:
            self.evidence_provenance = evidence

        self.state_history.append({
            "from_state": old_state.value,
            "to_state": new_state.value,
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert claim record to a serializable dictionary."""
        return {
            "claim_id": self.claim_id,
            "source_sentence_id": self.source_sentence_id,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "claim_type": self.claim_type.value,
            "decomposition_status": self.decomposition_status.value,
            "state": self.state.value,
            "state_history": self.state_history,
            "triage_score": self.triage_score,
            "triage_features": self.triage_features,
            "uncertainty_flag": self.uncertainty_flag,
            "is_substantive": self.is_substantive,
            "is_limitation": self.is_limitation,
            "evidence_provenance": self.evidence_provenance,
            "notes": self.notes
        }
