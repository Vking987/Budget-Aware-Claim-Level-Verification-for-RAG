"""Stage B & B2: Sequential Budget-Constrained Verification Controller package."""

from .budget import BudgetTracker
from .controller import VerificationController
from .interfaces import BaseRepairer, BaseRetriever, LLMRepairer, MockRepairer, MockRetriever
from .mab_policy import (
    ContextualMABPolicy,
    FEATURE_NAMES,
    LinUCBArm,
    MABPolicyConfig,
    POLICY_SCHEMA_VERSION,
    extract_context_features,
)
from .models import (
    ControllerAction,
    ControllerTrace,
    PerClaimBudget,
    RepairResult,
    StoppingReason,
    TraceStep,
    VerificationBudget,
)
from .policy import ControllerPolicyConfig, SequentialDecisionPolicy, StaticAllocationPolicy, StaticCTSRankedPolicy
from .retriever import QdrantRetriever
from .reward import (
    BaseRewardModel,
    CorrectnessObservability,
    GoldAnnotatedRewardModel,
    ProxyVerificationRewardModel,
    RewardOutcome,
)

__all__ = [
    "ControllerAction",
    "StoppingReason",
    "VerificationBudget",
    "PerClaimBudget",
    "RepairResult",
    "TraceStep",
    "ControllerTrace",
    "BudgetTracker",
    "BaseRetriever",
    "MockRetriever",
    "QdrantRetriever",
    "BaseRepairer",
    "MockRepairer",
    "LLMRepairer",
    "ControllerPolicyConfig",
    "SequentialDecisionPolicy",
    "StaticAllocationPolicy",
    "StaticCTSRankedPolicy",
    "VerificationController",
    "BaseRewardModel",
    "CorrectnessObservability",
    "RewardOutcome",
    "ProxyVerificationRewardModel",
    "GoldAnnotatedRewardModel",
    "ContextualMABPolicy",
    "MABPolicyConfig",
    "LinUCBArm",
    "FEATURE_NAMES",
    "POLICY_SCHEMA_VERSION",
    "extract_context_features",
]

