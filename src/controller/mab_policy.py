"""Contextual Multi-Armed Bandit (MAB) policy for sequential verification.

Implements a Disjoint Linear Upper Confidence Bound (LinUCB) contextual bandit:
- Disjoint linear models per action arm: RETRIEVE, VERIFY, REPAIR.
- Online recursive parameter updates via Sherman-Morrison rank-1 formulation.
- Explicit uncertainty bonus: UCB = x^T theta + alpha * sqrt(x^T A^{-1} x).
- Hard budget constraint enforcement: only affordable, legal arms are evaluated.
- Deterministic tie-breaking for perfect reproducibility.
- Full state serialization and multi-episode persistence.

CRITICAL METHODOLOGICAL PRINCIPLES:
1. LinUCB estimates expected quality reward and uncertainty per context-action pair.
2. CTS is treated as an uncalibrated contextual feature, NOT a truth probability.
3. High UCB score never overrides a hard budget constraint.
4. Policy state persists across multiple episodes/answers to enable genuine learning.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.claims.models import ClaimRecord, ClaimState, ClaimType
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost
from .budget import BudgetTracker
from .models import ControllerAction, RepairResult
from .reward import BaseRewardModel, ProxyVerificationRewardModel, RewardOutcome


# ------------------------------------------------------------------------------
# Dependency-Free Linear Algebra Helpers
# ------------------------------------------------------------------------------

def _dot_product(u: List[float], v: List[float]) -> float:
    return sum(x * y for x, y in zip(u, v))


def _matrix_vector_mul(M: List[List[float]], v: List[float]) -> List[float]:
    return [_dot_product(row, v) for row in M]


def _identity_matrix(d: int, scale: float = 1.0) -> List[List[float]]:
    return [[scale if i == j else 0.0 for j in range(d)] for i in range(d)]


def _sherman_morrison_update(A_inv: List[List[float]], x: List[float]) -> List[List[float]]:
    """Rank-1 Sherman-Morrison inverse update: (A + x x^T)^(-1).

    (A + x x^T)^(-1) = A^(-1) - (A^(-1) x x^T A^(-1)) / (1 + x^T A^(-1) x)
    """
    d = len(x)
    # v = A^(-1) x
    v = _matrix_vector_mul(A_inv, x)
    # denom = 1 + x^T v
    denom = 1.0 + _dot_product(x, v)
    if abs(denom) < 1e-12:
        return A_inv

    new_inv = [[A_inv[i][j] for j in range(d)] for i in range(d)]
    for i in range(d):
        for j in range(d):
            new_inv[i][j] -= (v[i] * v[j]) / denom
    return new_inv


# ------------------------------------------------------------------------------
# LinUCB Arm Model
# ------------------------------------------------------------------------------

class LinUCBArm:
    """Represents a single action arm parameterized by ridge regression."""

    def __init__(self, action: ControllerAction, dimension: int = 8, l2_reg: float = 1.0) -> None:
        self.action = action
        self.dimension = dimension
        self.l2_reg = l2_reg

        # Design matrix inverse (A^(-1)), initialized to (1 / l2_reg) * I
        self.A_inv: List[List[float]] = _identity_matrix(dimension, scale=1.0 / l2_reg)
        # Response vector b = sum(reward * x)
        self.b: List[float] = [0.0] * dimension
        # Update counter
        self.update_count: int = 0

    def get_theta(self) -> List[float]:
        """Compute current point estimate: theta_hat = A^(-1) b."""
        return _matrix_vector_mul(self.A_inv, self.b)

    def predict(self, x: List[float], alpha: float) -> Tuple[float, float, float]:
        """Compute expected reward, uncertainty bonus, and total UCB index.

        Returns:
            Tuple of (expected_mean, uncertainty_bonus, ucb_score).
        """
        theta = self.get_theta()
        mean = _dot_product(x, theta)

        # Variance: x^T A^(-1) x
        A_inv_x = _matrix_vector_mul(self.A_inv, x)
        variance = _dot_product(x, A_inv_x)
        std_dev = math.sqrt(max(0.0, variance))
        bonus = alpha * std_dev
        ucb = mean + bonus

        return mean, bonus, ucb

    def update(self, x: List[float], reward: float) -> None:
        """Update arm parameters with observed context vector and scalar reward."""
        if not math.isfinite(reward):
            raise ValueError(f"Reward must be finite, got {reward}")

        # Sherman-Morrison rank-1 update on A_inv
        self.A_inv = _sherman_morrison_update(self.A_inv, x)
        # Update b
        for i in range(self.dimension):
            self.b[i] += reward * x[i]
        self.update_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "dimension": self.dimension,
            "l2_reg": self.l2_reg,
            "A_inv": self.A_inv,
            "b": self.b,
            "update_count": self.update_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LinUCBArm:
        arm = cls(
            action=ControllerAction(data["action"]),
            dimension=data["dimension"],
            l2_reg=data["l2_reg"],
        )
        arm.A_inv = data["A_inv"]
        arm.b = data["b"]
        arm.update_count = data["update_count"]
        return arm


# ------------------------------------------------------------------------------
# Contextual Feature Extraction
# ------------------------------------------------------------------------------

def extract_context_features(
    claim: ClaimRecord,
    action: ControllerAction,
    evidence_store: Dict[str, List[EvidencePassage]],
    assessments: Dict[str, EvidenceAssessment],
    budget_tracker: BudgetTracker,
) -> List[float]:
    """Construct a dense normalized feature vector x in R^8 for a (claim, action) pair.

    Features:
    0: Bias intercept (1.0)
    1: Claim Triage Score (CTS in [0, 1], default 0.5)
    2: Claim type risk heuristic (Temporal=0.9, Numerical=0.85, Factual=0.75, Negation=0.65, Other=0.4)
    3: Evidence passage coverage (min(1.0, count / 3.0))
    4: Assessment status (-1.0 if contradicted/conflict, 1.0 if insufficient, 0.5 if supported, 0.0 if none)
    5: Claim interaction history (min(1.0, action_count / 3.0))
    6: Global budget fraction remaining (steps_remaining / max_steps)
    7: Action affinity indicator (structural alignment between state and action)
    """
    # 0: Intercept
    x0 = 1.0

    # 1: CTS
    x1 = claim.triage_score if claim.triage_score is not None else 0.5
    x1 = max(0.0, min(1.0, float(x1)))

    # 2: Type risk
    type_weights = {
        ClaimType.TEMPORAL_COMPARISON: 0.90,
        ClaimType.NUMERICAL_QUANTITATIVE: 0.85,
        ClaimType.FACTUAL_RELATION: 0.75,
        ClaimType.EXISTENCE_NEGATION: 0.65,
        ClaimType.ATTRIBUTION: 0.50,
        ClaimType.UNKNOWN_GENERAL: 0.40,
    }
    x2 = type_weights.get(claim.claim_type, 0.50)

    # 3: Evidence coverage
    passages = evidence_store.get(claim.claim_id, [])
    x3 = min(1.0, len(passages) / 3.0)

    # 4: Assessment status
    assessment = assessments.get(claim.claim_id)
    if assessment is None:
        x4 = 0.0
    elif assessment.label == AssessmentLabel.INSUFFICIENT:
        x4 = 1.0
    elif assessment.label in (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED):
        x4 = -1.0
    elif assessment.label == AssessmentLabel.SUPPORTED:
        x4 = 0.5
    else:
        x4 = 0.0

    # 5: Previous actions on this claim
    claim_cost = budget_tracker.get_claim_cost(claim.claim_id)
    total_acts = claim_cost.get("retrievals", 0) + claim_cost.get("verifications", 0) + claim_cost.get("repairs", 0)
    x5 = min(1.0, total_acts / 3.0)

    # 6: Global budget fraction remaining
    gb = budget_tracker.global_budget
    if gb.max_steps is not None and gb.max_steps > 0:
        rem_steps = max(0, gb.max_steps - budget_tracker.steps_taken)
        x6 = rem_steps / gb.max_steps
    else:
        x6 = 1.0

    # 7: Action-state affinity
    if action == ControllerAction.RETRIEVE:
        # High affinity if 0 passages or assessment was insufficient
        x7 = 1.0 if (not passages or (assessment and assessment.label == AssessmentLabel.INSUFFICIENT)) else 0.2
    elif action == ControllerAction.VERIFY:
        # High affinity if has passages but unassessed
        x7 = 1.0 if (passages and assessment is None) else 0.2
    elif action == ControllerAction.REPAIR:
        # High affinity if contradicted or conflicted
        x7 = 1.0 if (assessment and assessment.label in (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED)) else 0.0
    else:
        x7 = 0.0

    return [x0, x1, x2, x3, x4, x5, x6, x7]


# ------------------------------------------------------------------------------
# Contextual MAB Policy
# ------------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "bias",
    "claim_triage_score",
    "claim_type_risk",
    "evidence_coverage",
    "assessment_status",
    "interaction_history",
    "budget_fraction_left",
    "action_state_affinity",
]
POLICY_SCHEMA_VERSION = "1.0.0"


@dataclass
class MABPolicyConfig:
    """Configuration for Contextual LinUCB verification policy."""
    alpha: float = 0.50             # Exploration rate (0.0 = pure exploitation)
    l2_reg: float = 1.0             # Ridge regression regularization
    feature_dimension: int = 8      # Context vector dimension
    verification_confidence_threshold: float = 0.60
    enable_repair: bool = True
    allow_reconsideration: bool = True
    deterministic_tie_breaker: str = "claim_id"
    frozen: bool = False            # When True, prohibits parameter updates during evaluation
    random_seed: Optional[int] = None

    @property
    def verification_threshold(self) -> float:
        return self.verification_confidence_threshold

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



class ContextualMABPolicy:
    """Contextual multi-armed bandit policy using Disjoint LinUCB."""

    def __init__(
        self,
        config: Optional[MABPolicyConfig] = None,
        reward_model: Optional[BaseRewardModel] = None,
    ) -> None:
        self.config = config or MABPolicyConfig()
        self.reward_model = reward_model or ProxyVerificationRewardModel(
            verification_threshold=self.config.verification_confidence_threshold
        )
        self.frozen: bool = self.config.frozen

        dim = self.config.feature_dimension
        reg = self.config.l2_reg

        # Arms for candidate active verification actions
        self.arms: Dict[ControllerAction, LinUCBArm] = {
            ControllerAction.RETRIEVE: LinUCBArm(ControllerAction.RETRIEVE, dimension=dim, l2_reg=reg),
            ControllerAction.VERIFY: LinUCBArm(ControllerAction.VERIFY, dimension=dim, l2_reg=reg),
            ControllerAction.REPAIR: LinUCBArm(ControllerAction.REPAIR, dimension=dim, l2_reg=reg),
        }

        # Episode and interaction accounting
        self.episodes_trained: int = 0
        self.total_decisions: int = 0
        self.total_feedback_updates: int = 0

        # Checkpoint provenance tracking
        self.training_split: Optional[str] = None
        self.random_seed: Optional[int] = None
        self.checkpoint_metadata: Dict[str, Any] = {}

        # Last selected (claim_id, action, context_vector) awaiting feedback
        self._pending_context: Optional[Tuple[str, ControllerAction, List[float]]] = None

        # Observability only (never read by the policy): every candidate scored in the most recent select_next_action call
        self.last_decision: Optional[Dict[str, Any]] = None

    def freeze(self) -> None:
        """Freeze policy parameters for locked held-out evaluation."""
        self.frozen = True
        self.config.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze policy parameters to enable online learning."""
        self.frozen = False
        self.config.frozen = False

    def select_next_action(
        self,
        claims: List[ClaimRecord],
        evidence_store: Dict[str, List[EvidencePassage]],
        assessments: Dict[str, EvidenceAssessment],
        repairs: Dict[str, RepairResult],
        budget_tracker: BudgetTracker,
    ) -> Tuple[ControllerAction, Optional[ClaimRecord], str]:
        """Select next best claim-action arm based on LinUCB index and hard budget checks."""
        self.total_decisions += 1
        self.last_decision = None
        decision_log: List[Dict[str, Any]] = []

        # Hard Rule 1: Check global budget exhaustion
        if budget_tracker.is_exhausted():
            return (ControllerAction.STOP, None, "Global budget exhausted")

        if not claims:
            return (ControllerAction.STOP, None, "No claims provided")

        # Candidate pool: evaluate all legal, affordable claim-action pairs
        candidates = []

        for claim in claims:
            # Verified and Unresolved claims have concluded active investigation
            if claim.state in (ClaimState.VERIFIED, ClaimState.UNRESOLVED, ClaimState.DEFERRED):
                continue

            claim_id = claim.claim_id
            c_evidence = evidence_store.get(claim_id, [])
            c_assessment = assessments.get(claim_id)
            c_repaired = claim_id in repairs

            # Identify which actions are legally permissible under state and budget
            permissible_actions: List[ControllerAction] = []

            # RETRIEVE is permissible if: no evidence yet OR assessment was INSUFFICIENT
            if not c_evidence or (c_assessment and c_assessment.label == AssessmentLabel.INSUFFICIENT):
                # Use query-aware stall detection: permit retrieval when query changed or no prior attempt
                current_query = claim.normalized_text or claim.original_text
                if not budget_tracker.is_retrieval_stalled(claim_id, current_query=current_query):
                    can_ret, _ = budget_tracker.can_afford(ControllerAction.RETRIEVE, claim_id)
                    if can_ret:
                        permissible_actions.append(ControllerAction.RETRIEVE)

            # VERIFY is permissible if: has evidence and no assessment (or new evidence arrived)
            if c_evidence and c_assessment is None:
                can_ver, _ = budget_tracker.can_afford(ControllerAction.VERIFY, claim_id, estimated_model_calls=1)
                if can_ver:
                    permissible_actions.append(ControllerAction.VERIFY)

            # REPAIR is permissible if: contradicted/conflicted and not yet repaired
            if (
                self.config.enable_repair
                and c_assessment
                and c_assessment.label in (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED)
                and not c_repaired
            ):
                can_rep, _ = budget_tracker.can_afford(ControllerAction.REPAIR, claim_id)
                if can_rep:
                    permissible_actions.append(ControllerAction.REPAIR)

            # Score each permissible arm with LinUCB
            for act in permissible_actions:
                arm = self.arms.get(act)
                if arm is None:
                    continue

                x = extract_context_features(claim, act, evidence_store, assessments, budget_tracker)
                mean, bonus, ucb = arm.predict(x, alpha=self.config.alpha)

                # Deterministic tie-breaking key:
                # 1. -ucb (highest UCB first)
                # 2. -cts (higher CTS first)
                # 3. claim_id ascending
                triage = claim.triage_score if claim.triage_score is not None else 0.5
                sort_key = (-round(ucb, 6), -round(triage, 4), claim.claim_id, act.value)
                rationale = (
                    f"LinUCB selected arm {act.value} for claim '{claim_id}' "
                    f"(mu={mean:.3f}, bonus={bonus:.3f}, ucb={ucb:.3f})"
                )
                candidates.append((sort_key, act, claim, x, rationale))
                decision_log.append({"claim_id": claim_id, "action": act.value, "features": dict(zip(FEATURE_NAMES, x)),
                                     "mean": mean, "bonus": bonus, "ucb": ucb, "cts": triage})

        # If eligible candidate arms exist, select top candidate
        if candidates:
            candidates.sort(key=lambda item: item[0])
            _, best_action, best_claim, best_x, best_rationale = candidates[0]
            self.last_decision = {"alpha": self.config.alpha, "frozen": self.frozen, "candidates": decision_log,
                                  "chosen": {"claim_id": best_claim.claim_id, "action": best_action.value},
                                  "tie_break": "highest UCB, then higher CTS, then claim_id"}
            # Cache pending context for subsequent record_feedback call
            self._pending_context = (best_claim.claim_id, best_action, best_x)
            return best_action, best_claim, best_rationale

        # If no active claims, handle reconsideration of deferred claims if permitted
        if self.config.allow_reconsideration:
            deferred = [c for c in claims if c.state == ClaimState.DEFERRED]
            if deferred:
                # Pick highest CTS deferred claim to reconsider
                deferred.sort(key=lambda c: (-(c.triage_score or 0.5), c.claim_id))
                rec_claim = deferred[0]
                can_ret, _ = budget_tracker.can_afford(ControllerAction.RETRIEVE, rec_claim.claim_id)
                can_ver, _ = budget_tracker.can_afford(ControllerAction.VERIFY, rec_claim.claim_id)
                if can_ret or can_ver:
                    act = ControllerAction.RETRIEVE if not evidence_store.get(rec_claim.claim_id) else ControllerAction.VERIFY
                    x = extract_context_features(rec_claim, act, evidence_store, assessments, budget_tracker)
                    self._pending_context = (rec_claim.claim_id, act, x)
                    return act, rec_claim, f"MAB reconsidering deferred claim '{rec_claim.claim_id}'"

        # No actions remain
        self._pending_context = None
        return (
            ControllerAction.STOP,
            None,
            "All eligible arms completed or constrained by budget."
        )

    def record_feedback(
        self,
        claim: ClaimRecord,
        action: ControllerAction,
        new_passages: Optional[List[EvidencePassage]] = None,
        cumulative_passages: Optional[List[EvidencePassage]] = None,
        passages: Optional[List[EvidencePassage]] = None,
        evidence_passages: Optional[List[EvidencePassage]] = None,
        assessment: Optional[EvidenceAssessment] = None,
        repair: Optional[RepairResult] = None,
        repair_result: Optional[RepairResult] = None,
        cost: Optional[VerificationCost] = None,
        action_failed: bool = False,
        failure_reason: Optional[str] = None,
        explicit_reward: Optional[float] = None,
    ) -> Optional[float]:
        """Update LinUCB arm parameters from executed action outcome.

        Args:
            claim: Claim on which action was taken.
            action: Action executed.
            new_passages: Passages returned by the current retrieval action.
            cumulative_passages: Full evidence pool for this claim.
            passages: Backward-compatible alias for passages.
            evidence_passages: Backward-compatible alias for passages.
            assessment: Assessment produced if action was VERIFY.
            repair: Repair result produced if action was REPAIR.
            repair_result: Backward-compatible alias for repair.
            cost: Actual verification cost.
            action_failed: True if action threw an execution exception.
            failure_reason: Error message if action failed.
            explicit_reward: Optional pre-computed scalar reward (overrides reward_model).

        Returns:
            Scalar reward observed and used for the parameter update.
        """
        arm = self.arms.get(action)
        if arm is None:
            return None

        cum_p = cumulative_passages if cumulative_passages is not None else (passages or evidence_passages or [])
        new_p = new_passages if new_passages is not None else cum_p
        rep = repair if repair is not None else repair_result

        # Determine context vector used
        if self._pending_context and self._pending_context[0] == claim.claim_id and self._pending_context[1] == action:
            x = self._pending_context[2]
        else:
            # Reconstruct default context vector if not cached
            dummy_tracker = BudgetTracker()
            x = extract_context_features(claim, action, {claim.claim_id: cum_p}, {claim.claim_id: assessment} if assessment else {}, dummy_tracker)

        # Compute or validate reward
        if explicit_reward is not None:
            if not math.isfinite(explicit_reward):
                raise ValueError(f"Explicit reward must be finite, got {explicit_reward}")
            reward_val = max(-1.0, min(1.0, float(explicit_reward)))
        else:
            reward_outcome = self.reward_model.compute_reward(
                claim=claim,
                action=action,
                new_passages=new_p,
                cumulative_passages=cum_p,
                passages=cum_p,
                assessment=assessment,
                repair=rep,
                cost=cost,
                action_failed=action_failed,
                failure_reason=failure_reason,
            )
            reward_val = reward_outcome.quality_reward

        # If policy is frozen, clear pending context and return reward without mutating parameters
        if self.frozen:
            self._pending_context = None
            return reward_val

        # Update arm ridge regression
        arm.update(x, reward_val)
        self.total_feedback_updates += 1
        self._pending_context = None

        return reward_val

    def mark_episode_end(self, episode_id: Optional[str] = None) -> None:
        """Increment trained episode count.

        If an episode_id is provided, guards against duplicate finalization so
        the same episode cannot be double-counted or corrupt policy statistics (AUD-07).
        If policy is frozen, episode training counter is not modified.
        """
        if self.frozen:
            return

        if episode_id is not None:
            if not hasattr(self, "_completed_episode_ids"):
                self._completed_episode_ids = set()
            if episode_id in self._completed_episode_ids:
                return
            self._completed_episode_ids.add(episode_id)
        self.episodes_trained += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete policy hyperparameters and learned parameters."""
        return {
            "config": self.config.to_dict(),
            "frozen": self.frozen,
            "training_split": self.training_split,
            "random_seed": self.random_seed,
            "episodes_trained": self.episodes_trained,
            "total_decisions": self.total_decisions,
            "total_feedback_updates": self.total_feedback_updates,
            "arm_parameters": {act.value: arm.to_dict() for act, arm in self.arms.items()},
            "arms": {act.value: arm.to_dict() for act, arm in self.arms.items()},
            "metadata": dict(self.checkpoint_metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContextualMABPolicy:
        """Reconstruct trained policy from serialized dictionary."""
        config_data = data.get("config") or data.get("hyperparameters") or {}
        config = MABPolicyConfig(
            alpha=config_data.get("alpha", 0.50),
            l2_reg=config_data.get("l2_reg", 1.0),
            feature_dimension=config_data.get("feature_dimension", 8),
            verification_confidence_threshold=config_data.get(
                "verification_confidence_threshold",
                config_data.get("verification_threshold", 0.60),
            ),
            enable_repair=config_data.get("enable_repair", True),
            allow_reconsideration=config_data.get("allow_reconsideration", True),
            deterministic_tie_breaker=config_data.get("deterministic_tie_breaker", "claim_id"),
            frozen=config_data.get("frozen", data.get("frozen", False)),
            random_seed=data.get("random_seed", config_data.get("random_seed")),
        )
        policy = cls(config=config)
        policy.frozen = config.frozen
        policy.training_split = data.get("training_split")
        policy.random_seed = data.get("random_seed")
        policy.checkpoint_metadata = data.get("metadata", {})
        policy.episodes_trained = data.get("episodes_trained", 0)
        policy.total_decisions = data.get("total_decisions", 0)
        policy.total_feedback_updates = data.get("total_feedback_updates", 0)

        arms_dict = data.get("arms") or data.get("arm_parameters") or {}
        for act_str, arm_data in arms_dict.items():
            act = ControllerAction(act_str)
            policy.arms[act] = LinUCBArm.from_dict(arm_data)

        return policy

    def to_checkpoint(
        self,
        training_split: Optional[str] = None,
        random_seed: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produce reproducible checkpoint recording full provenance and hyperparameters."""
        split_id = training_split or self.training_split or "unspecified"
        seed_val = random_seed if random_seed is not None else self.random_seed
        meta = dict(self.checkpoint_metadata)
        meta["training_split"] = split_id
        meta["random_seed"] = seed_val
        if extra_metadata:
            meta.update(extra_metadata)

        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "policy_schema_version": POLICY_SCHEMA_VERSION,
            "policy_type": self.__class__.__name__,
            "policy_version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "training_split": split_id,
            "random_seed": seed_val,
            "hyperparameters": {
                "alpha": self.config.alpha,
                "l2_reg": self.config.l2_reg,
                "feature_dimension": self.config.feature_dimension,
                "verification_confidence_threshold": self.config.verification_confidence_threshold,
                "enable_repair": self.config.enable_repair,
                "allow_reconsideration": self.config.allow_reconsideration,
            },
            "feature_schema": {
                "dimension": self.config.feature_dimension,
                "feature_dimension": self.config.feature_dimension,
                "feature_names": list(FEATURE_NAMES),
            },
            "reward_model": {
                "identity": self.reward_model.__class__.__name__,
                "version": getattr(self.reward_model, "version", "1.0.0"),
            },
            "config": self.config.to_dict(),
            "frozen": self.frozen,
            "episodes_trained": self.episodes_trained,
            "total_decisions": self.total_decisions,
            "total_feedback_updates": self.total_feedback_updates,
            "arm_parameters": {act.value: arm.to_dict() for act, arm in self.arms.items()},
            "arms": {act.value: arm.to_dict() for act, arm in self.arms.items()},
            "metadata": meta,
        }

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        training_split: Optional[str] = None,
        random_seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save reproducible policy checkpoint to JSON file."""
        checkpoint_data = self.to_checkpoint(
            training_split=training_split,
            random_seed=random_seed,
            extra_metadata=metadata,
        )
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

    @classmethod
    def load_checkpoint(
        cls,
        filepath_or_dict: Union[str, Path, Dict[str, Any]],
        frozen: bool = True,
    ) -> ContextualMABPolicy:
        """Load and validate policy checkpoint with strict compatibility checks."""
        if isinstance(filepath_or_dict, (str, Path)):
            with open(filepath_or_dict, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(filepath_or_dict, dict):
            data = filepath_or_dict
        else:
            raise TypeError(f"Expected filepath or dict, got {type(filepath_or_dict)}")

        # 1. Validate policy type compatibility
        policy_type = data.get("policy_type")
        valid_policy_types = (cls.__name__, "LinUCB_Disjoint", "LinUCB")
        if policy_type and policy_type not in valid_policy_types:
            raise ValueError(
                f"Incompatible policy type: checkpoint contains '{policy_type}', expected one of {valid_policy_types}."
            )

        # 2. Validate feature schema dimension
        feat_schema = data.get("feature_schema", {})
        ckpt_dim = feat_schema.get("dimension", feat_schema.get("feature_dimension"))
        if ckpt_dim is not None:
            expected_dim = len(FEATURE_NAMES)
            if ckpt_dim != expected_dim:
                raise ValueError(
                    f"Feature dimension mismatch: checkpoint feature dimension {ckpt_dim} != expected {expected_dim}."
                )

        # 3. Validate feature names and ordering
        if "feature_names" in feat_schema:
            ckpt_names = feat_schema["feature_names"]
            if ckpt_names != FEATURE_NAMES:
                raise ValueError(
                    f"Feature name ordering mismatch: checkpoint features {ckpt_names} != expected {FEATURE_NAMES}."
                )

        # 4. Construct base policy using from_dict
        policy = cls.from_dict(data)

        # 5. Attach provenance metadata
        policy.training_split = data.get("training_split")
        policy.random_seed = data.get("random_seed")
        policy.checkpoint_metadata = data.get("metadata", {})

        # 6. Apply frozen evaluation state if requested
        if frozen:
            policy.freeze()

        return policy

    def save(self, filepath: Union[str, Path]) -> None:
        """Save policy state to JSON file."""
        self.save_checkpoint(filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> ContextualMABPolicy:
        """Load policy state from JSON file."""
        return cls.load_checkpoint(filepath, frozen=False)
