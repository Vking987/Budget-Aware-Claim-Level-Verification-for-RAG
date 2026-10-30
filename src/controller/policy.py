"""Sequential decision policy for Stage B verification controller.

Determines:
1. Which claim to investigate next.
2. Which action (RETRIEVE, VERIFY, REPAIR, DEFER, STOP) to apply.
3. Priority ranking based on Claim Triage Score (CTS), claim type, and dynamic state.
4. Deterministic tie-breaking for perfect reproducibility.

CRITICAL METHODOLOGICAL PRINCIPLE:
CTS is a heuristic prioritization signal, NOT a calibrated probability that
a claim is false. Higher CTS prioritizes investigation but does not prejudge
the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.claims.models import ClaimRecord, ClaimState, ClaimType
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage
from .budget import BudgetTracker
from .models import ControllerAction, RepairResult


@dataclass
class ControllerPolicyConfig:
    """Configurable hyperparameters for controller action selection."""
    verification_confidence_threshold: float = 0.60
    enable_repair: bool = True
    allow_reconsideration: bool = True
    max_retrieval_attempts_per_claim: int = 2
    max_verification_attempts_per_claim: int = 2
    min_triage_score_threshold: float = 0.0
    deterministic_tie_breaker: str = "claim_id"

    # Action priority weights for tie-breaking
    action_priority: Dict[ControllerAction, int] = None

    def __post_init__(self) -> None:
        if self.action_priority is None:
            self.action_priority = {
                ControllerAction.REPAIR: 4,
                ControllerAction.VERIFY: 3,
                ControllerAction.RETRIEVE: 2,
                ControllerAction.DEFER: 1,
                ControllerAction.STOP: 0,
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verification_confidence_threshold": self.verification_confidence_threshold,
            "enable_repair": self.enable_repair,
            "allow_reconsideration": self.allow_reconsideration,
            "max_retrieval_attempts_per_claim": self.max_retrieval_attempts_per_claim,
            "max_verification_attempts_per_claim": self.max_verification_attempts_per_claim,
            "min_triage_score_threshold": self.min_triage_score_threshold,
            "deterministic_tie_breaker": self.deterministic_tie_breaker,
        }


class SequentialDecisionPolicy:
    """Transparent, deterministic baseline policy for sequential verification."""

    def __init__(self, config: Optional[ControllerPolicyConfig] = None) -> None:
        self.config = config or ControllerPolicyConfig()

    def record_feedback(self, *args, **kwargs) -> None:
        """No-op for deterministic static heuristic baseline."""
        pass

    def select_next_action(
        self,
        claims: List[ClaimRecord],
        evidence_store: Dict[str, List[EvidencePassage]],
        assessments: Dict[str, EvidenceAssessment],
        repairs: Dict[str, RepairResult],
        budget_tracker: BudgetTracker,
    ) -> Tuple[ControllerAction, Optional[ClaimRecord], str]:
        """Evaluate current system state and select the single next best action.

        Returns:
            Tuple of (selected_action, target_claim_record, selection_rationale).
        """
        # Rule 1: Check global budget exhaustion first
        if budget_tracker.is_exhausted():
            return (
                ControllerAction.STOP,
                None,
                "Global verification budget is exhausted."
            )

        if not claims:
            return (
                ControllerAction.STOP,
                None,
                "No claims provided to controller."
            )

        # Check if all claims have reached terminal outcomes
        all_resolved = True
        for c in claims:
            if c.state not in (ClaimState.VERIFIED, ClaimState.UNRESOLVED, ClaimState.DEFERRED):
                all_resolved = False
                break

        # Candidate pool 1: Active, unresolved claims (PENDING, SELECTED, UNRESOLVED with remaining attempts)
        candidates: List[Tuple[Tuple[float, int, int, str], ControllerAction, ClaimRecord, str]] = []

        for claim in claims:
            claim_id = claim.claim_id
            state = claim.state

            # Rule: Claims in terminal/concluded states are not in the primary candidate pool
            if state in (ClaimState.VERIFIED, ClaimState.UNRESOLVED, ClaimState.DEFERRED):
                continue

            c_evidence = evidence_store.get(claim_id, [])
            c_assessment = assessments.get(claim_id)
            c_repaired = claim_id in repairs

            # Determine eligible action for this claim
            action = None
            rationale = ""

            # Scenario 1: No evidence yet -> RETRIEVE
            if not c_evidence:
                can_ret, ret_msg = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim_id)
                if can_ret:
                    action = ControllerAction.RETRIEVE
                    rationale = f"Claim '{claim_id}' has no candidate evidence; initiating retrieval."
                else:
                    # Cannot afford retrieval -> DEFER
                    action = ControllerAction.DEFER
                    rationale = f"Cannot afford retrieval for claim '{claim_id}' ({ret_msg}); deferring."

            # Scenario 2: Has evidence but no assessment -> VERIFY
            elif c_assessment is None:
                can_ver, ver_msg = budget_tracker.can_afford(ControllerAction.VERIFY, claim_id, estimated_model_calls=1)
                if can_ver:
                    action = ControllerAction.VERIFY
                    rationale = f"Claim '{claim_id}' has {len(c_evidence)} passages but no assessment; verifying."
                else:
                    action = ControllerAction.DEFER
                    rationale = f"Cannot afford verification for claim '{claim_id}' ({ver_msg}); deferring."

            # Scenario 3: Has assessment -> check outcome
            else:
                label = c_assessment.label

                if label == AssessmentLabel.SUPPORTED:
                    # If supported with sufficient confidence, it's ready for VERIFIED transition
                    if c_assessment.confidence >= self.config.verification_confidence_threshold:
                        # Already supported with high confidence, no further action needed
                        continue
                    else:
                        # Low-confidence support -> attempt further retrieval if allowed
                        can_ret, _ = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim_id)
                        if can_ret:
                            action = ControllerAction.RETRIEVE
                            rationale = (
                                f"Claim '{claim_id}' is SUPPORTED with low confidence "
                                f"({c_assessment.confidence:.2f} < {self.config.verification_confidence_threshold}); "
                                f"retrieving additional evidence."
                            )
                        else:
                            # Cannot retrieve more -> leave as UNRESOLVED
                            continue

                elif label in (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED):
                    if self.config.enable_repair and not c_repaired:
                        can_rep, rep_msg = budget_tracker.can_afford(ControllerAction.REPAIR, claim_id)
                        if can_rep:
                            action = ControllerAction.REPAIR
                            rationale = (
                                f"Claim '{claim_id}' was assessed as {label.value}; "
                                f"initiating structured repair/qualification."
                            )
                        else:
                            continue
                    else:
                        # Already repaired or repair disabled -> terminal contested claim
                        continue

                elif label == AssessmentLabel.INSUFFICIENT:
                    claim_cost = budget_tracker.get_claim_cost(claim_id)
                    had_prior_retrieval = claim_cost.get("retrievals", 0) > 0
                    last_unique = budget_tracker.last_retrieval_unique_count(claim_id)
                    if had_prior_retrieval and last_unique == 0:
                        continue
                    # Inconclusive evidence -> retrieve more if budget and limits permit
                    can_ret, ret_msg = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim_id)
                    if can_ret:
                        action = ControllerAction.RETRIEVE
                        rationale = (
                            f"Previous assessment was INSUFFICIENT for claim '{claim_id}'; "
                            f"requesting additional evidence passages."
                        )
                    else:
                        # Cannot retrieve more -> leaves claim as UNRESOLVED, no endless loop
                        continue

            if action is not None:
                # Build deterministic sorting key:
                # 1. triage_score descending (None treated as 0.0)
                # 2. claim_type risk weight descending
                # 3. action priority descending
                # 4. claim_id ascending (tie-breaker)
                triage = claim.triage_score if claim.triage_score is not None else 0.0
                type_rank = 1 if claim.claim_type == ClaimType.TEMPORAL_COMPARISON else 0
                act_rank = self.config.action_priority.get(action, 0)
                # Invert numbers for max-heap via standard min-sort
                sort_key = (-triage, -type_rank, -act_rank, claim.claim_id)
                candidates.append((sort_key, action, claim, rationale))

        # If we have eligible candidates, pick the top one
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, best_action, best_claim, best_rationale = candidates[0]
            return best_action, best_claim, best_rationale

        # If no candidates among active claims, check if any DEFERRED claims can be reconsidered
        if self.config.allow_reconsideration:
            deferred_candidates = []
            for claim in claims:
                if claim.state == ClaimState.DEFERRED:
                    # Can we afford retrieval or verification now?
                    can_ret, _ = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim.claim_id)
                    can_ver, _ = budget_tracker.can_afford(ControllerAction.VERIFY, claim.claim_id)
                    if can_ret or can_ver:
                        triage = claim.triage_score if claim.triage_score is not None else 0.0
                        deferred_candidates.append((
                            (-triage, claim.claim_id),
                            claim
                        ))
            if deferred_candidates:
                deferred_candidates.sort(key=lambda x: x[0])
                _, reconsidered_claim = deferred_candidates[0]
                # If it has no evidence, retrieve; otherwise verify
                c_evidence = evidence_store.get(reconsidered_claim.claim_id, [])
                if not c_evidence:
                    return (
                        ControllerAction.RETRIEVE,
                        reconsidered_claim,
                        f"Reconsidering previously deferred claim '{reconsidered_claim.claim_id}' under remaining budget."
                    )
                else:
                    return (
                        ControllerAction.VERIFY,
                        reconsidered_claim,
                        f"Reconsidering verification for previously deferred claim '{reconsidered_claim.claim_id}'."
                    )

        # Nothing left to do
        return (
            ControllerAction.STOP,
            None,
            "All eligible claims have been resolved or cannot be further investigated within budget."
        )


class StaticAllocationPolicy:
    """First-In, First-Out (FIFO) baseline policy for sequential verification.

    Allocates verification actions strictly in claim-arrival order.
    Does NOT use Claim Triage Score (CTS), claim type weighting, or learned rewards.
    Operates under the exact same budget constraints, action eligibility checks,
    and state transitions as SequentialDecisionPolicy and ContextualMABPolicy.
    """

    def __init__(self, config: Optional[ControllerPolicyConfig] = None) -> None:
        self.config = config or ControllerPolicyConfig()

    def record_feedback(self, *args, **kwargs) -> None:
        """No-op for static baseline policy."""
        pass

    def select_next_action(
        self,
        claims: List[ClaimRecord],
        evidence_store: Dict[str, List[EvidencePassage]],
        assessments: Dict[str, EvidenceAssessment],
        repairs: Dict[str, RepairResult],
        budget_tracker: BudgetTracker,
    ) -> Tuple[ControllerAction, Optional[ClaimRecord], str]:
        """Select next action strictly following claim arrival sequence."""
        if budget_tracker.is_exhausted():
            return ControllerAction.STOP, None, "Global verification budget is exhausted."

        if not claims:
            return ControllerAction.STOP, None, "No claims provided to controller."

        # Scan claims strictly in arrival order (FIFO)
        for claim in claims:
            claim_id = claim.claim_id
            state = claim.state

            # Concluded or terminal claims are skipped
            if state in (ClaimState.VERIFIED, ClaimState.UNRESOLVED, ClaimState.DEFERRED):
                continue

            c_evidence = evidence_store.get(claim_id, [])
            c_assessment = assessments.get(claim_id)
            c_repaired = claim_id in repairs

            # 1. No evidence yet -> RETRIEVE (or DEFER if unaffordable)
            if not c_evidence:
                can_ret, ret_msg = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim_id)
                if can_ret:
                    return (
                        ControllerAction.RETRIEVE,
                        claim,
                        f"FIFO: Claim '{claim_id}' has no candidate evidence; initiating retrieval."
                    )
                else:
                    return (
                        ControllerAction.DEFER,
                        claim,
                        f"FIFO: Cannot afford retrieval for claim '{claim_id}' ({ret_msg}); deferring."
                    )

            # 2. Has evidence but no assessment -> VERIFY (or DEFER if unaffordable)
            elif c_assessment is None:
                can_ver, ver_msg = budget_tracker.can_afford(ControllerAction.VERIFY, claim_id, estimated_model_calls=1)
                if can_ver:
                    return (
                        ControllerAction.VERIFY,
                        claim,
                        f"FIFO: Claim '{claim_id}' has {len(c_evidence)} passages but no assessment; verifying."
                    )
                else:
                    return (
                        ControllerAction.DEFER,
                        claim,
                        f"FIFO: Cannot afford verification for claim '{claim_id}' ({ver_msg}); deferring."
                    )

            # 3. Has assessment
            else:
                label = c_assessment.label

                if label == AssessmentLabel.SUPPORTED:
                    if c_assessment.confidence >= self.config.verification_confidence_threshold:
                        continue  # Already supported with high confidence
                    else:
                        can_ret, _ = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim_id)
                        if can_ret:
                            return (
                                ControllerAction.RETRIEVE,
                                claim,
                                f"FIFO: Claim '{claim_id}' SUPPORTED with low confidence; retrieving additional evidence."
                            )
                        else:
                            continue

                elif label in (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED):
                    if self.config.enable_repair and not c_repaired:
                        can_rep, _ = budget_tracker.can_afford(ControllerAction.REPAIR, claim_id)
                        if can_rep:
                            return (
                                ControllerAction.REPAIR,
                                claim,
                                f"FIFO: Claim '{claim_id}' was assessed as {label.value}; initiating repair."
                            )
                        else:
                            continue
                    else:
                        continue

                elif label == AssessmentLabel.INSUFFICIENT:
                    claim_cost = budget_tracker.get_claim_cost(claim_id)
                    had_prior_retrieval = claim_cost.get("retrievals", 0) > 0
                    last_unique = budget_tracker.last_retrieval_unique_count(claim_id)
                    if had_prior_retrieval and last_unique == 0:
                        continue
                    can_ret, _ = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim_id)
                    if can_ret:
                        return (
                            ControllerAction.RETRIEVE,
                            claim,
                            f"FIFO: Claim '{claim_id}' assessment INSUFFICIENT; retrieving additional evidence."
                        )
                    else:
                        continue

        # If all active claims are processed or unaffordable, check reconsideration of deferred claims in FIFO order
        if self.config.allow_reconsideration:
            for claim in claims:
                if claim.state == ClaimState.DEFERRED:
                    can_ret, _ = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim.claim_id)
                    can_ver, _ = budget_tracker.can_afford(ControllerAction.VERIFY, claim.claim_id)
                    if can_ret or can_ver:
                        c_evidence = evidence_store.get(claim.claim_id, [])
                        if not c_evidence and can_ret:
                            return (
                                ControllerAction.RETRIEVE,
                                claim,
                                f"FIFO: Reconsidering deferred claim '{claim.claim_id}' for retrieval."
                            )
                        elif c_evidence and can_ver:
                            return (
                                ControllerAction.VERIFY,
                                claim,
                                f"FIFO: Reconsidering deferred claim '{claim.claim_id}' for verification."
                            )

        return (
            ControllerAction.STOP,
            None,
            "FIFO: All claims have been processed or budget is exhausted."
        )


class StaticCTSRankedPolicy:
    """Static CTS-Ranked baseline policy for sequential verification.

    Methodological Role (AUD-04):
    Isolates the contribution of dynamic reallocation vs. static prioritization.
    - Sorts all eligible claims ONCE at the start of an episode using the Claim Triage Score (CTS).
    - Processes claims top-to-bottom according to this fixed ranking.
    - Does NOT dynamically reorder remaining claims based on newly observed verification outcomes.
    - Uses identical action eligibility checks, budget enforcement, and state transitions
      as StaticAllocationPolicy (FIFO) and SequentialDecisionPolicy.
    """

    def __init__(self, config: Optional[ControllerPolicyConfig] = None) -> None:
        self.config = config or ControllerPolicyConfig()
        self._fixed_ranking: Optional[List[str]] = None
        self.initial_ranking: List[str] = []

    def record_feedback(self, *args, **kwargs) -> None:
        """No-op for static baseline policy."""
        pass

    def reset_ranking(self) -> None:
        """Reset the cached ranking for a new episode."""
        self._fixed_ranking = None
        self.initial_ranking = []

    def select_next_action(
        self,
        claims: List[ClaimRecord],
        evidence_store: Dict[str, List[EvidencePassage]],
        assessments: Dict[str, EvidenceAssessment],
        repairs: Dict[str, RepairResult],
        budget_tracker: BudgetTracker,
    ) -> Tuple[ControllerAction, Optional[ClaimRecord], str]:
        """Select next action following a fixed static CTS ranking determined once at start."""
        if budget_tracker.is_exhausted():
            return ControllerAction.STOP, None, "Global verification budget is exhausted."

        if not claims:
            return ControllerAction.STOP, None, "No claims provided to controller."

        # Rank once at initialization (or if new set of claims provided)
        claim_map = {c.claim_id: c for c in claims}
        if self._fixed_ranking is None or set(self._fixed_ranking) != set(claim_map.keys()):
            # Deterministic sorting key:
            # 1. -triage_score (None treated as 0.0)
            # 2. -type_rank (1 if TEMPORAL_COMPARISON else 0)
            # 3. claim_id ascending (deterministic tie-breaker)
            sorted_claims = sorted(
                claims,
                key=lambda c: (
                    -(c.triage_score if c.triage_score is not None else 0.0),
                    -(1 if c.claim_type == ClaimType.TEMPORAL_COMPARISON else 0),
                    c.claim_id,
                )
            )
            self._fixed_ranking = [c.claim_id for c in sorted_claims]
            self.initial_ranking = list(self._fixed_ranking)

        # Process claims strictly in the fixed static CTS order
        for claim_id in self._fixed_ranking:
            claim = claim_map.get(claim_id)
            if claim is None:
                continue

            state = claim.state

            # Concluded or terminal claims are skipped
            if state in (ClaimState.VERIFIED, ClaimState.UNRESOLVED, ClaimState.DEFERRED):
                continue

            c_evidence = evidence_store.get(claim_id, [])
            c_assessment = assessments.get(claim_id)
            c_repaired = claim_id in repairs
            triage_val = claim.triage_score if claim.triage_score is not None else 0.0

            # 1. No evidence yet -> RETRIEVE (or DEFER if unaffordable)
            if not c_evidence:
                can_ret, ret_msg = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim_id)
                if can_ret:
                    return (
                        ControllerAction.RETRIEVE,
                        claim,
                        f"StaticCTS: Claim '{claim_id}' (CTS={triage_val:.2f}) has no candidate evidence; initiating retrieval."
                    )
                else:
                    return (
                        ControllerAction.DEFER,
                        claim,
                        f"StaticCTS: Cannot afford retrieval for claim '{claim_id}' ({ret_msg}); deferring."
                    )

            # 2. Has evidence but no assessment -> VERIFY (or DEFER if unaffordable)
            elif c_assessment is None:
                can_ver, ver_msg = budget_tracker.can_afford(ControllerAction.VERIFY, claim_id, estimated_model_calls=1)
                if can_ver:
                    return (
                        ControllerAction.VERIFY,
                        claim,
                        f"StaticCTS: Claim '{claim_id}' (CTS={triage_val:.2f}) has {len(c_evidence)} passages but no assessment; verifying."
                    )
                else:
                    return (
                        ControllerAction.DEFER,
                        claim,
                        f"StaticCTS: Cannot afford verification for claim '{claim_id}' ({ver_msg}); deferring."
                    )

            # 3. Has assessment
            else:
                label = c_assessment.label

                if label == AssessmentLabel.SUPPORTED:
                    if c_assessment.confidence >= self.config.verification_confidence_threshold:
                        continue  # Already supported with high confidence
                    else:
                        can_ret, _ = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim_id)
                        if can_ret:
                            return (
                                ControllerAction.RETRIEVE,
                                claim,
                                f"StaticCTS: Claim '{claim_id}' SUPPORTED with low confidence; retrieving additional evidence."
                            )
                        else:
                            continue

                elif label in (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED):
                    if self.config.enable_repair and not c_repaired:
                        can_rep, _ = budget_tracker.can_afford(ControllerAction.REPAIR, claim_id)
                        if can_rep:
                            return (
                                ControllerAction.REPAIR,
                                claim,
                                f"StaticCTS: Claim '{claim_id}' was assessed as {label.value}; initiating repair."
                            )
                        else:
                            continue
                    else:
                        continue

                elif label == AssessmentLabel.INSUFFICIENT:
                    claim_cost = budget_tracker.get_claim_cost(claim_id)
                    had_prior_retrieval = claim_cost.get("retrievals", 0) > 0
                    last_unique = budget_tracker.last_retrieval_unique_count(claim_id)
                    if had_prior_retrieval and last_unique == 0:
                        continue
                    can_ret, _ = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim_id)
                    if can_ret:
                        return (
                            ControllerAction.RETRIEVE,
                            claim,
                            f"StaticCTS: Claim '{claim_id}' assessment INSUFFICIENT; retrieving additional evidence."
                        )
                    else:
                        continue

        # Reconsider deferred claims strictly in fixed CTS order
        if self.config.allow_reconsideration:
            for claim_id in self._fixed_ranking:
                claim = claim_map.get(claim_id)
                if claim and claim.state == ClaimState.DEFERRED:
                    can_ret, _ = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim_id)
                    can_ver, _ = budget_tracker.can_afford(ControllerAction.VERIFY, claim_id)
                    if can_ret or can_ver:
                        c_evidence = evidence_store.get(claim_id, [])
                        if not c_evidence and can_ret:
                            return (
                                ControllerAction.RETRIEVE,
                                claim,
                                f"StaticCTS: Reconsidering deferred claim '{claim_id}' for retrieval."
                            )
                        elif c_evidence and can_ver:
                            return (
                                ControllerAction.VERIFY,
                                claim,
                                f"StaticCTS: Reconsidering deferred claim '{claim_id}' for verification."
                            )

        return (
            ControllerAction.STOP,
            None,
            "StaticCTS: All claims processed or budget exhausted."
        )

