"""Reward models and evaluation interfaces for contextual verification bandits.

Defines modular reward interfaces separate from the decision policy:
- BaseRewardModel: Abstract interface for computing scalar feedback rewards.
- RewardOutcome: Dataclass capturing quality reward, risk reduction, costs, provenance,
  and explicit observability flags.
- ProxyVerificationRewardModel: Runtime reward heuristic based on marginal evidence grounding,
  contradiction discovery, and repair progress (noting assessor noise).
- GoldAnnotatedRewardModel: Offline evaluation reward comparing retrieved evidence
  and claim alignment against ground-truth supporting facts.

CRITICAL METHODOLOGICAL PRINCIPLES:
1. Quality reward is strictly separated from resource consumption and hard budget limits.
2. A gold document title match alone NEVER proves that an extracted claim is factually correct.
3. Unobservable claim-level correctness is explicitly distinguished from true negative reward.
4. Marginal evidence acquisition is evaluated on novel passages, preventing repeated rewards
   for stale or duplicate evidence.
5. Failed actions receive explicit failure penalties and are never awarded positive success rewards.
"""

from __future__ import annotations

import math
import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set

from src.claims.models import ClaimRecord, ClaimState
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost
from .models import ControllerAction, RepairResult


class CorrectnessObservability(str, Enum):
    """Explicit observability status for claim-level ground truth."""
    OBSERVABLE_CORRECT = "OBSERVABLE_CORRECT"
    OBSERVABLE_INCORRECT = "OBSERVABLE_INCORRECT"
    UNOBSERVABLE = "UNOBSERVABLE"


@dataclass
class RewardOutcome:
    """Structured reward signal returned after action execution."""
    quality_reward: float
    evidence_retrieval_reward: Optional[float] = None
    claim_correctness_reward: Optional[float] = None
    correctness_observability: CorrectnessObservability = CorrectnessObservability.UNOBSERVABLE
    is_action_failure: bool = False
    heuristic_risk_reduction: float = 0.0
    cost: Optional[VerificationCost] = None
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.quality_reward):
            raise ValueError(f"quality_reward must be a finite float, got {self.quality_reward}")
        # Clamp reward to [-1.0, 1.0] interval
        self.quality_reward = max(-1.0, min(1.0, float(self.quality_reward)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_reward": round(self.quality_reward, 4),
            "evidence_retrieval_reward": (
                round(self.evidence_retrieval_reward, 4)
                if self.evidence_retrieval_reward is not None else None
            ),
            "claim_correctness_reward": (
                round(self.claim_correctness_reward, 4)
                if self.claim_correctness_reward is not None else None
            ),
            "correctness_observability": self.correctness_observability.value,
            "is_action_failure": self.is_action_failure,
            "heuristic_risk_reduction": round(self.heuristic_risk_reduction, 4),
            "cost": self.cost.to_dict() if self.cost else None,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }

    @property
    def reward_provenance(self) -> str:
        return self.rationale

    @property
    def is_factual_correctness_verified(self) -> bool:
        return (
            self.correctness_observability == CorrectnessObservability.OBSERVABLE_CORRECT
            and self.claim_correctness_reward is not None
            and self.claim_correctness_reward > 0.0
        )



def _normalize_text(text: str) -> str:
    """Standard QA text normalization (lowercase, remove punctuation and articles)."""
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _get_passage_key(passage: EvidencePassage) -> str:
    """Compute stable identifier for evidence deduplication."""
    if passage.evidence_id and passage.evidence_id.strip():
        return passage.evidence_id.strip()
    return f"{passage.doc_title.lower().strip()}::{passage.source_text.strip()[:100]}"


class BaseRewardModel(ABC):
    """Abstract interface for bandit verification reward models."""

    @abstractmethod
    def compute_reward(
        self,
        claim: ClaimRecord,
        action: ControllerAction,
        new_passages: Optional[List[EvidencePassage]] = None,
        cumulative_passages: Optional[List[EvidencePassage]] = None,
        assessment: Optional[EvidenceAssessment] = None,
        repair: Optional[RepairResult] = None,
        cost: Optional[VerificationCost] = None,
        action_failed: bool = False,
        failure_reason: Optional[str] = None,
        passages: Optional[List[EvidencePassage]] = None,
    ) -> RewardOutcome:
        """Compute scalar feedback reward for an executed claim-action pair."""
        ...


class ProxyVerificationRewardModel(BaseRewardModel):
    """Runtime heuristic reward model based on observable verification transitions.

    Evaluates marginal evidence novelty, contradiction discovery, and qualification progress.
    """

    def __init__(self, verification_threshold: float = 0.60) -> None:
        self.verification_threshold = verification_threshold

    def compute_reward(
        self,
        claim: ClaimRecord,
        action: ControllerAction,
        new_passages: Optional[List[EvidencePassage]] = None,
        cumulative_passages: Optional[List[EvidencePassage]] = None,
        assessment: Optional[EvidenceAssessment] = None,
        repair: Optional[RepairResult] = None,
        cost: Optional[VerificationCost] = None,
        action_failed: bool = False,
        failure_reason: Optional[str] = None,
        passages: Optional[List[EvidencePassage]] = None,
    ) -> RewardOutcome:
        # Fallback for backward compatibility
        cum_passages = cumulative_passages if cumulative_passages is not None else (passages or [])
        marginal_passages = new_passages if new_passages is not None else cum_passages
        heuristic_risk = claim.triage_score if claim.triage_score is not None else 0.5

        # Rule 1: Explicit Action Failure Handling
        if action_failed or (assessment is not None and getattr(assessment, "is_error", False)):
            effective_reason = failure_reason or (assessment.rationale if assessment else "Unknown")
            return RewardOutcome(
                quality_reward=-0.8,
                is_action_failure=True,
                cost=cost,
                rationale=f"Action {action.value} failed: {effective_reason}",
                metadata={"action_failed": True, "failure_reason": effective_reason},
            )

        # Rule 2: RETRIEVE - Marginal novelty evaluation
        if action == ControllerAction.RETRIEVE:
            if not marginal_passages:
                return RewardOutcome(
                    quality_reward=-0.3,
                    evidence_retrieval_reward=-0.3,
                    heuristic_risk_reduction=0.0,
                    cost=cost,
                    rationale="Unproductive retrieval: 0 new passages returned.",
                )

            # Check if all marginal passages were already in cumulative pool prior to action
            prior_count = len(cum_passages) - len(marginal_passages)
            prior_keys = {
                _get_passage_key(p) for p in cum_passages[:max(0, prior_count)]
            }
            novel_passages = [p for p in marginal_passages if _get_passage_key(p) not in prior_keys]

            if not novel_passages:
                return RewardOutcome(
                    quality_reward=-0.2,
                    evidence_retrieval_reward=-0.2,
                    heuristic_risk_reduction=0.0,
                    cost=cost,
                    rationale="Repeated retrieval returned only duplicate passages already present in evidence pool.",
                )

            return RewardOutcome(
                quality_reward=0.2,
                evidence_retrieval_reward=0.2,
                heuristic_risk_reduction=0.05 * heuristic_risk,
                cost=cost,
                rationale=f"Successfully retrieved {len(novel_passages)} novel candidate evidence passages.",
                metadata={"novel_passage_count": len(novel_passages)},
            )

        # Rule 3: VERIFY
        elif action == ControllerAction.VERIFY:
            if assessment is None:
                return RewardOutcome(
                    quality_reward=-0.5,
                    is_action_failure=True,
                    cost=cost,
                    rationale="Verification action produced no assessment object.",
                )

            label = assessment.label
            conf = assessment.confidence

            if label == AssessmentLabel.SUPPORTED:
                if conf >= self.verification_threshold:
                    risk_reduction = 0.8 * heuristic_risk
                    return RewardOutcome(
                        quality_reward=0.8,
                        heuristic_risk_reduction=risk_reduction,
                        cost=cost,
                        rationale=f"Claim substantiated by evidence with high confidence (conf={conf:.2f}).",
                    )
                else:
                    return RewardOutcome(
                        quality_reward=0.1,
                        heuristic_risk_reduction=0.1 * heuristic_risk,
                        cost=cost,
                        rationale=f"Claim weakly supported below verification threshold (conf={conf:.2f}).",
                    )

            elif label in (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED):
                risk_reduction = 0.6 * heuristic_risk
                return RewardOutcome(
                    quality_reward=0.6,
                    heuristic_risk_reduction=risk_reduction,
                    cost=cost,
                    rationale=f"Discovered factual contradiction ({label.value}) preventing silent hallucination.",
                )

            elif label == AssessmentLabel.INSUFFICIENT:
                return RewardOutcome(
                    quality_reward=-0.1,
                    heuristic_risk_reduction=0.0,
                    cost=cost,
                    rationale="Evidence remains inconclusive/insufficient.",
                )

        # Rule 4: REPAIR
        elif action == ControllerAction.REPAIR:
            if repair is None:
                return RewardOutcome(
                    quality_reward=-0.5,
                    is_action_failure=True,
                    cost=cost,
                    rationale="Repair action failed to produce a valid revision.",
                )
            risk_reduction = 0.7 * heuristic_risk
            return RewardOutcome(
                quality_reward=0.7,
                heuristic_risk_reduction=risk_reduction,
                cost=cost,
                rationale=f"Contradicted claim qualified: {repair.proposed_revision}",
            )

        return RewardOutcome(
            quality_reward=0.0,
            cost=cost,
            rationale=f"Control action {action.value} had neutral quality impact.",
        )


class GoldAnnotatedRewardModel(BaseRewardModel):
    """Offline evaluation reward model using gold HotpotQA supporting facts.

    Maintains a strict scientific separation between:
    1. Evidence Retrieval Reward: Whether newly retrieved passages match gold supporting document titles.
    2. Claim Correctness Reward: Whether the claim's asserted facts agree with or contradict
       available ground truth answer annotations.
    """

    def __init__(
        self,
        gold_supporting_titles: Optional[Iterable[str]] = None,
        gold_answer: str = "",
        gold_titles: Optional[Iterable[str]] = None,
    ) -> None:
        titles = gold_supporting_titles if gold_supporting_titles is not None else (gold_titles or set())
        self.gold_supporting_titles = {t.lower().strip() for t in titles}
        self.gold_answer = gold_answer.strip()
        self.normalized_gold_answer = _normalize_text(gold_answer)

    def compute_reward(
        self,
        claim: ClaimRecord,
        action: ControllerAction,
        new_passages: Optional[List[EvidencePassage]] = None,
        cumulative_passages: Optional[List[EvidencePassage]] = None,
        assessment: Optional[EvidenceAssessment] = None,
        repair: Optional[RepairResult] = None,
        cost: Optional[VerificationCost] = None,
        action_failed: bool = False,
        failure_reason: Optional[str] = None,
        passages: Optional[List[EvidencePassage]] = None,
    ) -> RewardOutcome:
        cum_passages = cumulative_passages if cumulative_passages is not None else (passages or [])
        marginal_passages = new_passages if new_passages is not None else cum_passages

        # Action Failure Handling
        if action_failed or (assessment is not None and getattr(assessment, "is_error", False)):
            effective_reason = failure_reason or (assessment.rationale if assessment else "Unknown")
            return RewardOutcome(
                quality_reward=-0.8,
                is_action_failure=True,
                cost=cost,
                rationale=f"Action failed with exception: {effective_reason}",
            )

        # RETRIEVE Action
        if action == ControllerAction.RETRIEVE:
            if not marginal_passages:
                return RewardOutcome(
                    quality_reward=-0.3,
                    evidence_retrieval_reward=-0.3,
                    correctness_observability=CorrectnessObservability.UNOBSERVABLE,
                    cost=cost,
                    rationale="Unproductive retrieval: 0 passages returned.",
                )

            # Deduplicate against prior cumulative passages
            prior_count = len(cum_passages) - len(marginal_passages)
            prior_keys = {
                _get_passage_key(p) for p in cum_passages[:max(0, prior_count)]
            }
            novel_passages = [p for p in marginal_passages if _get_passage_key(p) not in prior_keys]

            if not novel_passages:
                return RewardOutcome(
                    quality_reward=-0.2,
                    evidence_retrieval_reward=-0.2,
                    correctness_observability=CorrectnessObservability.UNOBSERVABLE,
                    cost=cost,
                    rationale="Repeated retrieval returned only duplicate passages already present in evidence pool.",
                )

            # Evaluate whether novel passages match gold supporting titles
            novel_titles = {p.doc_title.lower().strip() for p in novel_passages}
            matched_gold = novel_titles & self.gold_supporting_titles

            if matched_gold:
                return RewardOutcome(
                    quality_reward=1.0,
                    evidence_retrieval_reward=1.0,
                    correctness_observability=CorrectnessObservability.UNOBSERVABLE,
                    cost=cost,
                    rationale=f"Retrieved novel gold supporting documents: {matched_gold}",
                    metadata={"novel_gold_titles": list(matched_gold)},
                )
            else:
                return RewardOutcome(
                    quality_reward=0.1,
                    evidence_retrieval_reward=0.1,
                    correctness_observability=CorrectnessObservability.UNOBSERVABLE,
                    cost=cost,
                    rationale="Retrieved novel passages, but none matched gold supporting document titles.",
                )

        # VERIFY Action
        elif action == ControllerAction.VERIFY:
            if assessment is None:
                return RewardOutcome(
                    quality_reward=-0.5,
                    is_action_failure=True,
                    cost=cost,
                    rationale="Verification produced no assessment object.",
                )

            cum_titles = {p.doc_title.lower().strip() for p in cum_passages}
            has_gold_context = bool(cum_titles & self.gold_supporting_titles)

            # Evaluate claim-level ground truth alignment against gold_answer
            norm_claim = _normalize_text(claim.normalized_text or claim.original_text)
            norm_answer = self.normalized_gold_answer

            # Check if claim directly asserts or contradicts the gold answer string
            is_answer_bearing = bool(norm_answer and (norm_answer in norm_claim or norm_claim in norm_answer))

            # Case 1: Direct Answer-bearing claim
            if is_answer_bearing:
                if assessment.label == AssessmentLabel.SUPPORTED and has_gold_context:
                    return RewardOutcome(
                        quality_reward=1.0,
                        evidence_retrieval_reward=1.0 if has_gold_context else 0.5,
                        claim_correctness_reward=1.0,
                        correctness_observability=CorrectnessObservability.OBSERVABLE_CORRECT,
                        cost=cost,
                        rationale="Claim directly asserts gold answer and is substantiated by gold evidence context.",
                        metadata={"gold_answer": self.gold_answer},
                    )
                elif assessment.label == AssessmentLabel.CONTRADICTED:
                    # Claim asserted gold answer but assessor falsely contradicted it
                    return RewardOutcome(
                        quality_reward=-0.5,
                        claim_correctness_reward=-0.5,
                        correctness_observability=CorrectnessObservability.OBSERVABLE_CORRECT,
                        cost=cost,
                        rationale="Claim asserts gold answer, but assessor flagged a contradiction.",
                    )

            # Case 2: Claim explicitly asserts an entity/figure that contradicts the gold answer
            # E.g., if gold_answer is a 4-digit year or number that differs from numbers asserted in claim
            claim_years = re.findall(r"\b(?:1[7-9][0-9]{2}|20[0-2][0-9])\b", norm_claim)
            answer_years = re.findall(r"\b(?:1[7-9][0-9]{2}|20[0-2][0-9])\b", norm_answer)
            if answer_years and claim_years and set(answer_years) != set(claim_years):
                if assessment.label == AssessmentLabel.CONTRADICTED:
                    return RewardOutcome(
                        quality_reward=0.7,
                        claim_correctness_reward=0.7,
                        correctness_observability=CorrectnessObservability.OBSERVABLE_INCORRECT,
                        cost=cost,
                        rationale=f"Correctly flagged factual contradiction against gold answer '{self.gold_answer}'.",
                    )
                elif assessment.label == AssessmentLabel.SUPPORTED:
                    # Assessor hallucinated support for a claim that contradicts the gold answer
                    return RewardOutcome(
                        quality_reward=-0.7,
                        claim_correctness_reward=-0.7,
                        correctness_observability=CorrectnessObservability.OBSERVABLE_INCORRECT,
                        cost=cost,
                        rationale=f"Assessor supported claim that contradicts gold answer '{self.gold_answer}'.",
                    )

            # Case 3: Intermediate Bridging Claim (Correctness unobservable from gold answer alone)
            # A gold document title match alone does NOT prove factual correctness of bridge claims.
            if assessment.label == AssessmentLabel.SUPPORTED:
                q_rew = 0.5 if has_gold_context else 0.2
                return RewardOutcome(
                    quality_reward=q_rew,
                    evidence_retrieval_reward=1.0 if has_gold_context else 0.2,
                    claim_correctness_reward=None,
                    correctness_observability=CorrectnessObservability.UNOBSERVABLE,
                    cost=cost,
                    rationale=(
                        "Claim supported by evidence in gold context, but claim-level ground truth "
                        "is unobservable from final answer annotation alone."
                    ),
                    metadata={"has_gold_context": has_gold_context},
                )
            elif assessment.label in (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED):
                return RewardOutcome(
                    quality_reward=0.4,
                    evidence_retrieval_reward=0.5 if has_gold_context else 0.0,
                    claim_correctness_reward=None,
                    correctness_observability=CorrectnessObservability.UNOBSERVABLE,
                    cost=cost,
                    rationale="Contradiction flagged in intermediate claim context.",
                )
            else:
                return RewardOutcome(
                    quality_reward=-0.1,
                    evidence_retrieval_reward=0.0,
                    claim_correctness_reward=None,
                    correctness_observability=CorrectnessObservability.UNOBSERVABLE,
                    cost=cost,
                    rationale="Assessment inconclusive (INSUFFICIENT).",
                )

        # REPAIR Action
        elif action == ControllerAction.REPAIR:
            if repair:
                return RewardOutcome(
                    quality_reward=0.6,
                    cost=cost,
                    rationale="Repair executed and documented.",
                )

        return RewardOutcome(
            quality_reward=0.0,
            cost=cost,
            rationale="Neutral outcome.",
        )
