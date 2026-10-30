"""Sequential Verification Controller for Stage B.

Coordinates the iterative, budget-constrained claim verification loop:
1. Dynamically selects actions using SequentialDecisionPolicy.
2. Dispatches actions to Retriever, Assessor, and Repairer interfaces.
3. Audits all computational resources using BudgetTracker.
4. Manages legal ClaimState lifecycle transitions without bypassing invariants.
5. Emits complete, serializable decision traces for auditability and experiment tracking.

CRITICAL INVARIANTS:
1. Assessment SUPPORTED != ClaimState.VERIFIED.
2. Transition to VERIFIED requires an evidence payload and is only permitted
   when assessor confidence satisfies policy thresholds.
3. Repaired and deferred claims remain UNVERIFIED. Deferral is never evidence
   of correctness.
4. Baseline files, Qdrant vectors, and baseline evaluation harnesses remain untouched.
"""

from __future__ import annotations

import copy
import logging
import uuid
from typing import Any, Dict, List, Optional, Set

from src.claims.models import ClaimRecord, ClaimState, InvalidStateTransitionError
from src.evidence.assessor import BaseAssessor
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost
from .budget import BudgetTracker
from .interfaces import BaseRepairer, BaseRetriever, MockRepairer, MockRetriever
from .models import (
    ControllerAction,
    ControllerTrace,
    PerClaimBudget,
    RepairResult,
    StoppingReason,
    TraceStep,
    VerificationBudget,
)
from .policy import ControllerPolicyConfig, SequentialDecisionPolicy


class VerificationController:
    """Sequential resource allocation controller for claim verification."""

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        assessor: Optional[BaseAssessor] = None,
        repairer: Optional[BaseRepairer] = None,
        policy: Optional[SequentialDecisionPolicy] = None,
        global_budget: Optional[VerificationBudget] = None,
        per_claim_budget: Optional[PerClaimBudget] = None,
    ) -> None:
        self.retriever = retriever or MockRetriever()
        self.assessor = assessor
        self.repairer = repairer or MockRepairer()
        self.feedback_failures = 0  # count of swallowed policy.record_feedback errors
        self.policy = policy or SequentialDecisionPolicy()
        self.global_budget = global_budget or VerificationBudget()
        self.per_claim_budget = per_claim_budget or PerClaimBudget()

    def run(
        self,
        claims: List[ClaimRecord],
        initial_evidence: Optional[Dict[str, List[EvidencePassage]]] = None,
        initial_assessments: Optional[Dict[str, EvidenceAssessment]] = None,
        run_id: Optional[str] = None,
    ) -> ControllerTrace:
        """Execute the sequential verification loop over a list of claims.

        Args:
            claims: Input atomic claim propositions.
            initial_evidence: Pre-existing candidate evidence passages per claim ID.
            initial_assessments: Pre-existing evidence assessments per claim ID.
            run_id: Unique identifier for trace provenance.

        Returns:
            ControllerTrace containing complete decision history, cost breakdown,
            and final summary metrics.

        Raises:
            ValueError: If duplicate claim IDs or invalid budgets are passed.
        """
        run_identifier = run_id or f"run_{uuid.uuid4().hex[:12]}"

        # Contract validation: Duplicate claim IDs
        seen_ids: Set[str] = set()
        for c in claims:
            if c.claim_id in seen_ids:
                raise ValueError(f"Duplicate claim ID detected: '{c.claim_id}'. Claim IDs must be unique.")
            seen_ids.add(c.claim_id)

        # Initialize tracking stores
        evidence_store: Dict[str, List[EvidencePassage]] = {
            c.claim_id: list(initial_evidence.get(c.claim_id, []))
            for c in claims
        } if initial_evidence else {c.claim_id: [] for c in claims}

        assessments: Dict[str, EvidenceAssessment] = dict(initial_assessments) if initial_assessments else {}
        repairs: Dict[str, RepairResult] = {}
        budget_tracker = BudgetTracker(self.global_budget, self.per_claim_budget)
        trace_steps: List[TraceStep] = []

        # Handle zero budget or empty claims immediately
        if not claims:
            return ControllerTrace(
                run_id=run_identifier,
                policy_name=self.policy.__class__.__name__,
                policy_config=self.policy.config.to_dict(),
                initial_budget=self.global_budget.to_dict(),
                final_budget_remaining=budget_tracker.get_remaining_budget(),
                total_cost=budget_tracker.get_global_cost(),
                total_steps=0,
                stopping_reason=StoppingReason.NO_ELIGIBLE_ACTIONS,
                claims_summary={
                    "total": 0,
                    "verified": 0,
                    "unresolved": 0,
                    "deferred": 0,
                    "pending": 0,
                },
                steps=[],
            )

        stopping_reason = StoppingReason.NO_ELIGIBLE_ACTIONS

        while True:
            # 1. Ask policy for the next best action
            action, target_claim, rationale = self.policy.select_next_action(
                claims=claims,
                evidence_store=evidence_store,
                assessments=assessments,
                repairs=repairs,
                budget_tracker=budget_tracker,
            )

            # 2. Check stopping conditions
            if action == ControllerAction.STOP:
                if budget_tracker.is_exhausted():
                    stopping_reason = StoppingReason.BUDGET_EXHAUSTED
                else:
                    all_terminal = all(
                        c.state in (ClaimState.VERIFIED, ClaimState.UNRESOLVED, ClaimState.DEFERRED)
                        for c in claims
                    )
                    any_unresolved = any(c.state == ClaimState.UNRESOLVED for c in claims)
                    if all_terminal and not any_unresolved:
                        # Every claim reached a positive terminal state (VERIFIED or DEFERRED)
                        stopping_reason = StoppingReason.ALL_CLAIMS_RESOLVED
                    elif all_terminal:
                        # All done, but some claims ended UNRESOLVED — do NOT claim full resolution
                        stopping_reason = StoppingReason.NO_ELIGIBLE_ACTIONS
                    else:
                        stopping_reason = StoppingReason.NO_ELIGIBLE_ACTIONS

                # Record STOP step
                trace_steps.append(TraceStep(
                    step_number=budget_tracker.steps_taken + 1,
                    claim_id=None,
                    claim_text=None,
                    triage_score=None,
                    previous_state=None,
                    new_state=None,
                    action=ControllerAction.STOP,
                    selection_rationale=rationale,
                    action_outcome=f"Loop terminated: {stopping_reason.value}",
                    action_cost={},
                    cumulative_global_cost=budget_tracker.get_global_cost(),
                    per_claim_cumulative_cost={},
                    remaining_budget=budget_tracker.get_remaining_budget(),
                ))
                break

            # Process claim action
            claim_id = target_claim.claim_id
            prev_state = target_claim.state.value
            step_num = budget_tracker.steps_taken + 1
            action_cost_dict: Dict[str, Any] = {}
            action_outcome_msg = ""
            assessment_label = None
            assessment_confidence = None

            action_failed = False
            failure_reason = None
            new_passages: List[EvidencePassage] = []
            cost = None

            try:
                # -------------------------------------------------------------
                # Action A: DEFER
                # -------------------------------------------------------------
                if action == ControllerAction.DEFER:
                    if target_claim.state in (ClaimState.PENDING, ClaimState.SELECTED):
                        target_claim.transition_to(
                            ClaimState.DEFERRED,
                            reason=f"Deferred by controller: {rationale}"
                        )
                    budget_tracker.record_action(action, claim_id, cost=None)
                    action_outcome_msg = f"Claim '{claim_id}' deferred unverified."

                # -------------------------------------------------------------
                # Action B: RETRIEVE
                # -------------------------------------------------------------
                elif action == ControllerAction.RETRIEVE:
                    # Legal transition to SELECTED before investigating
                    if target_claim.state in (ClaimState.PENDING, ClaimState.DEFERRED, ClaimState.UNRESOLVED):
                        target_claim.transition_to(
                            ClaimState.SELECTED,
                            reason=f"Selected for evidence retrieval: {rationale}"
                        )

                    passages, cost = self.retriever.retrieve(target_claim)
                    
                    # Deduplicate against existing passages in evidence_store[claim_id]
                    existing_passage_ids = {p.evidence_id for p in evidence_store.get(claim_id, [])}
                    new_passages = [p for p in passages if p.evidence_id not in existing_passage_ids]
                    dup_passages = [p for p in passages if p.evidence_id in existing_passage_ids]

                    evidence_store[claim_id].extend(new_passages)

                    # Record unique vs duplicate passages in budget tracker
                    budget_tracker.record_retrieval_passages(claim_id, len(new_passages), len(dup_passages))
                    budget_tracker.record_action(action, claim_id, cost)
                    action_cost_dict = cost.to_dict()

                    if len(new_passages) > 0:
                        # Invalidate prior assessment ONLY if new evidence actually arrived
                        assessments.pop(claim_id, None)
                        action_outcome_msg = (
                            f"Retrieved {len(new_passages)} new unique passages "
                            f"({len(dup_passages)} duplicates) for claim '{claim_id}'."
                        )
                    else:
                        # Zero new passages retrieved -> no evidence progress made
                        action_outcome_msg = (
                            f"Retrieved 0 new unique passages ({len(dup_passages)} duplicates); "
                            f"evidence state unchanged for claim '{claim_id}'."
                        )
                        # If previously assessed as INSUFFICIENT, redundant retrieval indicates stalled progress
                        if claim_id in assessments and assessments[claim_id].label == AssessmentLabel.INSUFFICIENT:
                            target_claim.transition_to(
                                ClaimState.UNRESOLVED,
                                reason="Evidence INSUFFICIENT and redundant retrieval yielded no new passages (progress stalled)."
                            )
                            action_outcome_msg += " Transitioned to UNRESOLVED (progress stalled)."

                # -------------------------------------------------------------
                # Action C: VERIFY
                # -------------------------------------------------------------
                elif action == ControllerAction.VERIFY:
                    if target_claim.state in (ClaimState.PENDING, ClaimState.DEFERRED, ClaimState.UNRESOLVED):
                        target_claim.transition_to(
                            ClaimState.SELECTED,
                            reason=f"Selected for alignment verification: {rationale}"
                        )

                    passages = evidence_store.get(claim_id, [])
                    if self.assessor is None:
                        raise RuntimeError("Cannot execute VERIFY: No assessor backend configured on controller.")

                    assessment = self.assessor.assess(target_claim, passages)
                    assessments[claim_id] = assessment
                    assessment_label = assessment.label.value
                    assessment_confidence = assessment.confidence

                    cost = assessment.cost
                    budget_tracker.record_action(action, claim_id, assessment.cost)
                    action_cost_dict = assessment.cost.to_dict()
                    action_outcome_msg = f"Assessed as {assessment_label} with confidence {assessment.confidence:.2f}."

                    # Lifecycle state transition check
                    if getattr(assessment, "is_error", False):
                        action_failed = True
                        failure_reason = f"Assessor technical error: {assessment.rationale}"
                        target_claim.transition_to(
                            ClaimState.UNRESOLVED,
                            reason=failure_reason,
                        )
                        action_outcome_msg += f" [TECHNICAL ERROR: {assessment.error_type or 'Unknown'}] Transitioned to UNRESOLVED."

                    elif assessment.label == AssessmentLabel.SUPPORTED:
                        if assessment.confidence >= self.policy.config.verification_confidence_threshold:
                            # Legal transition to VERIFIED with evidence payload
                            target_claim.transition_to(
                                ClaimState.VERIFIED,
                                reason=(
                                    f"Assessor confirmed SUPPORTED with confidence "
                                    f"{assessment.confidence:.2f} >= {self.policy.config.verification_confidence_threshold}"
                                ),
                                evidence=assessment.to_dict(),
                            )
                            action_outcome_msg += " Claim transitioned to VERIFIED."
                        else:
                            action_outcome_msg += " Confidence below verification threshold; remains SELECTED."

                    elif assessment.label in (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED):
                        if not self.policy.config.enable_repair:
                            target_claim.transition_to(
                                ClaimState.UNRESOLVED,
                                reason=f"Assessment yielded {assessment.label.value}; repair disabled."
                            )
                            action_outcome_msg += " Repair disabled; claim transitioned to UNRESOLVED."

                    elif assessment.label == AssessmentLabel.INSUFFICIENT:
                        # If retrieval budget exhausted, claim cannot be substantiated
                        can_ret, _ = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim_id)
                        if not can_ret:
                            target_claim.transition_to(
                                ClaimState.UNRESOLVED,
                                reason="Evidence INSUFFICIENT and retrieval limit exhausted."
                            )
                            action_outcome_msg += " Retrieval limit exhausted; transitioned to UNRESOLVED."

                # -------------------------------------------------------------
                # Action D: REPAIR
                # -------------------------------------------------------------
                elif action == ControllerAction.REPAIR:
                    curr_assessment = assessments.get(claim_id)
                    passages = evidence_store.get(claim_id, [])
                    repair_res, cost = self.repairer.repair(target_claim, curr_assessment, passages)
                    repairs[claim_id] = repair_res

                    budget_tracker.record_action(action, claim_id, cost)
                    action_cost_dict = cost.to_dict()

                    # Repaired claim transitions to UNRESOLVED with documented repair notes
                    target_claim.transition_to(
                        ClaimState.UNRESOLVED,
                        reason=f"Claim qualified/repaired: {repair_res.proposed_revision}"
                    )
                    action_outcome_msg = f"Claim repaired: '{repair_res.proposed_revision}'. Transitioned to UNRESOLVED."

                    # Re-verify the revised text against the same evidence, if budget allows.
                    revision = (repair_res.proposed_revision or "").strip()
                    if revision and self.assessor is not None:
                        can_rv, rv_why = budget_tracker.can_afford(
                            ControllerAction.VERIFY, claim_id, estimated_model_calls=1
                        )
                        if can_rv:
                            revised_claim = copy.deepcopy(target_claim)
                            revised_claim.original_text = revision
                            revised_claim.normalized_text = revision
                            rv = self.assessor.assess(revised_claim, passages)
                            budget_tracker.record_action(ControllerAction.VERIFY, claim_id, rv.cost)
                            rv_ok = (
                                not getattr(rv, "is_error", False)
                                and rv.label == AssessmentLabel.SUPPORTED
                                and rv.confidence >= self.policy.config.verification_confidence_threshold
                            )
                            repair_res.reverification = {
                                "label": rv.label.value,
                                "confidence": rv.confidence,
                                "verified": rv_ok,
                                "cost": rv.cost.to_dict(),
                            }
                            action_outcome_msg += (
                                f" Re-verification: {rv.label.value} ({rv.confidence:.2f}), "
                                f"{'passed' if rv_ok else 'not verified'}."
                            )
                        else:
                            action_outcome_msg += f" Re-verification skipped: {rv_why}."

            except Exception as exc:
                action_failed = True
                failure_reason = str(exc)
                # Catch recoverable action failures without terminating the loop silently
                action_outcome_msg = f"ACTION FAILED with exception: {type(exc).__name__}: {str(exc)}"
                budget_tracker.record_action(action, claim_id, cost=None)
                if target_claim.state == ClaimState.SELECTED:
                    target_claim.transition_to(
                        ClaimState.UNRESOLVED,
                        reason=f"Action execution error: {str(exc)}"
                    )

            # Record feedback to adaptive policy (e.g., LinUCB)
            if hasattr(self.policy, "record_feedback"):
                try:
                    self.policy.record_feedback(
                        claim=target_claim,
                        action=action,
                        new_passages=new_passages if (not action_failed and action == ControllerAction.RETRIEVE) else [],
                        cumulative_passages=evidence_store.get(claim_id, []),
                        passages=new_passages if (not action_failed and action == ControllerAction.RETRIEVE) else evidence_store.get(claim_id, []),
                        assessment=assessments.get(claim_id) if not action_failed else None,
                        repair=repairs.get(claim_id) if not action_failed else None,
                        cost=cost if not action_failed and cost else None,
                        action_failed=action_failed,
                        failure_reason=failure_reason,
                    )
                except Exception as fb_exc:
                    self.feedback_failures += 1
                    logging.getLogger(__name__).warning(
                        "policy.record_feedback failed (%s): %s", type(fb_exc).__name__, fb_exc
                    )

            # Capture step-level artifacts for faithful post-hoc trace evaluation
            step_retrieved_dicts = [
                {
                    "evidence_id": p.evidence_id,
                    "doc_title": p.doc_title,
                    "source_text": p.source_text,
                    "retrieval_score": p.retrieval_score,
                    "metadata": dict(p.metadata) if p.metadata else {},
                }
                for p in new_passages
            ] if (action == ControllerAction.RETRIEVE and not action_failed) else []

            step_assessment_dict = assessments.get(claim_id).to_dict() if (action == ControllerAction.VERIFY and not action_failed and claim_id in assessments) else None
            step_repair_dict = repairs.get(claim_id).to_dict() if (action == ControllerAction.REPAIR and not action_failed and claim_id in repairs) else None

            # Record detailed trace step
            trace_steps.append(TraceStep(
                step_number=step_num,
                claim_id=claim_id,
                claim_text=target_claim.original_text,
                triage_score=target_claim.triage_score,
                previous_state=prev_state,
                new_state=target_claim.state.value,
                action=action,
                selection_rationale=rationale,
                action_outcome=action_outcome_msg,
                error=failure_reason if action_failed else None,
                assessment_label=assessment_label,
                assessment_confidence=assessment_confidence,
                action_cost=action_cost_dict,
                cumulative_global_cost=budget_tracker.get_global_cost(),
                per_claim_cumulative_cost=budget_tracker.get_claim_cost(claim_id),
                remaining_budget=budget_tracker.get_remaining_budget(),
                retrieved_passages=step_retrieved_dicts,
                assessment_data=step_assessment_dict,
                repair_data=step_repair_dict,
            ))

            # Check max steps limit
            if (
                self.global_budget.max_steps is not None
                and budget_tracker.steps_taken >= self.global_budget.max_steps
            ):
                stopping_reason = StoppingReason.MAX_STEPS_REACHED
                trace_steps.append(TraceStep(
                    step_number=budget_tracker.steps_taken + 1,
                    claim_id=None,
                    claim_text=None,
                    triage_score=None,
                    previous_state=None,
                    new_state=None,
                    action=ControllerAction.STOP,
                    selection_rationale=f"Global maximum steps ({self.global_budget.max_steps}) reached.",
                    action_outcome=f"Loop terminated: {stopping_reason.value}",
                    action_cost={},
                    cumulative_global_cost=budget_tracker.get_global_cost(),
                    per_claim_cumulative_cost={},
                    remaining_budget=budget_tracker.get_remaining_budget(),
                ))
                break

        # Mark episode completion on adaptive policies
        if hasattr(self.policy, "mark_episode_end"):
            import inspect
            sig = inspect.signature(self.policy.mark_episode_end)
            if len(sig.parameters) > 0:
                self.policy.mark_episode_end(run_identifier)
            else:
                self.policy.mark_episode_end()

        # Summary of claim states at termination
        verified_cnt = sum(1 for c in claims if c.state == ClaimState.VERIFIED)
        unresolved_cnt = sum(1 for c in claims if c.state == ClaimState.UNRESOLVED)
        deferred_cnt = sum(1 for c in claims if c.state == ClaimState.DEFERRED)
        pending_cnt = sum(1 for c in claims if c.state == ClaimState.PENDING)

        return ControllerTrace(
            run_id=run_identifier,
            policy_name=self.policy.__class__.__name__,
            policy_config=self.policy.config.to_dict(),
            initial_budget=self.global_budget.to_dict(),
            final_budget_remaining=budget_tracker.get_remaining_budget(),
            total_cost=budget_tracker.get_global_cost(),
            total_steps=budget_tracker.steps_taken,
            stopping_reason=stopping_reason,
            claims_summary={
                "total": len(claims),
                "verified": verified_cnt,
                "unresolved": unresolved_cnt,
                "deferred": deferred_cnt,
                "pending": pending_cnt,
            },
            steps=trace_steps,
        )
