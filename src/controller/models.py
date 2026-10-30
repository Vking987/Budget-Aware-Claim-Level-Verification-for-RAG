"""Data contracts, enumerations, and trace schemas for Stage B controller."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.evidence.models import VerificationCost


class ControllerAction(str, Enum):
    """Explicit actions selected by the verification controller.

    - RETRIEVE: Fetch additional evidence passages for a claim.
    - VERIFY: Evaluate alignment between claim and available evidence using an assessor.
    - REPAIR: Generate a proposed correction/qualification for a contradicted/conflicted claim.
    - DEFER: Postpone investigation of a claim due to budget or priority limits.
    - STOP: Terminate sequential investigation loop.
    """
    RETRIEVE = "RETRIEVE"
    VERIFY = "VERIFY"
    REPAIR = "REPAIR"
    DEFER = "DEFER"
    STOP = "STOP"


class StoppingReason(str, Enum):
    """Explicit termination conditions for controller execution."""
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    ALL_CLAIMS_RESOLVED = "ALL_CLAIMS_RESOLVED"
    NO_ELIGIBLE_ACTIONS = "NO_ELIGIBLE_ACTIONS"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    UNPRODUCTIVE_LOOP = "UNPRODUCTIVE_LOOP"
    MANUAL_HALT = "MANUAL_HALT"


@dataclass
class VerificationBudget:
    """Configurable global resource budget across distinct dimensions.

    NOTE: Model calls, tokens, and wall-clock latency are distinct units
    and must NEVER be summed into a single blended scalar.
    """
    max_model_calls: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None
    max_latency_seconds: Optional[float] = None
    max_steps: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_model_calls": self.max_model_calls,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_latency_seconds": self.max_latency_seconds,
            "max_steps": self.max_steps,
        }


@dataclass
class PerClaimBudget:
    """Resource constraints imposed per individual claim proposition."""
    max_retrievals: int = 2
    max_verifications: int = 2
    max_repairs: int = 1
    max_model_calls: Optional[int] = None
    max_total_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_retrievals": self.max_retrievals,
            "max_verifications": self.max_verifications,
            "max_repairs": self.max_repairs,
            "max_model_calls": self.max_model_calls,
            "max_total_tokens": self.max_total_tokens,
        }


@dataclass
class RepairResult:
    """Structured output from a claim repair action."""
    claim_id: str
    original_text: str
    proposed_revision: str
    repair_reason: str
    supporting_evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    cost: Optional[VerificationCost] = None
    # Outcome of re-assessing the revised text against the claim's evidence.
    # None = not attempted (e.g. budget exhausted); otherwise a dict with
    # label / confidence / verified / cost.
    reverification: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "original_text": self.original_text,
            "proposed_revision": self.proposed_revision,
            "repair_reason": self.repair_reason,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "confidence": self.confidence,
            "cost": self.cost.to_dict() if self.cost else None,
            "reverification": dict(self.reverification) if self.reverification else None,
        }


@dataclass
class TraceStep:
    """Detailed record of a single controller decision and execution step."""
    step_number: int
    claim_id: Optional[str]
    claim_text: Optional[str]
    triage_score: Optional[float]
    previous_state: Optional[str]
    new_state: Optional[str]
    action: ControllerAction
    selection_rationale: str
    action_outcome: Optional[str] = None
    error: Optional[str] = None
    assessment_label: Optional[str] = None
    assessment_confidence: Optional[float] = None
    action_cost: Dict[str, Any] = field(default_factory=dict)
    cumulative_global_cost: Dict[str, Any] = field(default_factory=dict)
    per_claim_cumulative_cost: Dict[str, Any] = field(default_factory=dict)
    remaining_budget: Dict[str, Any] = field(default_factory=dict)
    retrieved_passages: List[Dict[str, Any]] = field(default_factory=list)
    assessment_data: Optional[Dict[str, Any]] = None
    repair_data: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "triage_score": self.triage_score,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "action": self.action.value,
            "selection_rationale": self.selection_rationale,
            "action_outcome": self.action_outcome,
            "error": self.error,
            "assessment_label": self.assessment_label,
            "assessment_confidence": self.assessment_confidence,
            "action_cost": dict(self.action_cost),
            "cumulative_global_cost": dict(self.cumulative_global_cost),
            "per_claim_cumulative_cost": dict(self.per_claim_cumulative_cost),
            "remaining_budget": dict(self.remaining_budget),
            "retrieved_passages": list(self.retrieved_passages),
            "assessment_data": self.assessment_data,
            "repair_data": self.repair_data,
            "timestamp": self.timestamp,
        }


@dataclass
class ControllerTrace:
    """Complete, serializable trace of a controller verification run."""
    run_id: str
    policy_name: str
    policy_config: Dict[str, Any]
    initial_budget: Dict[str, Any]
    final_budget_remaining: Dict[str, Any]
    total_cost: Dict[str, Any]
    total_steps: int
    stopping_reason: StoppingReason
    claims_summary: Dict[str, int]
    steps: List[TraceStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "policy_name": self.policy_name,
            "policy_config": self.policy_config,
            "initial_budget": self.initial_budget,
            "final_budget_remaining": self.final_budget_remaining,
            "total_cost": self.total_cost,
            "total_steps": self.total_steps,
            "stopping_reason": self.stopping_reason.value,
            "claims_summary": dict(self.claims_summary),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
