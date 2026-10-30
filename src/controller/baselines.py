"""Additional budget-comparison baselines: seeded Random and breadth-first Verify-All.

Both obey the same eligibility rules as the existing policies (budget checks via
BudgetTracker.can_afford, terminal-state skipping, repair only after a CONTRADICTED /
CONFLICTED verdict, at most one repair per claim). They differ only in *which* eligible
(claim, action) pair is chosen next:

- RandomPolicy:    uniform over all eligible VERIFY/REPAIR pairs (seeded, reproducible).
- VerifyAllPolicy: verify every claim once in arrival order before spending any budget on
                   repair; repairs then go in arrival order with whatever budget remains.
                   (FIFO, by contrast, repairs claim i before verifying claim i+1.)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from src.claims.models import ClaimRecord, ClaimState
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage

from .budget import BudgetTracker
from .models import ControllerAction, RepairResult
from .policy import ControllerPolicyConfig

_TERMINAL = (ClaimState.VERIFIED, ClaimState.UNRESOLVED, ClaimState.DEFERRED)


class _EligibilityBaseline:
    def __init__(self, config: Optional[ControllerPolicyConfig] = None) -> None:
        self.config = config or ControllerPolicyConfig()

    def record_feedback(self, *args, **kwargs) -> None:
        """No-op: baseline does not learn."""

    def _eligible(
        self,
        claims: List[ClaimRecord],
        evidence_store: Dict[str, List[EvidencePassage]],
        assessments: Dict[str, EvidenceAssessment],
        repairs: Dict[str, RepairResult],
        bt: BudgetTracker,
    ) -> Tuple[List[ClaimRecord], List[ClaimRecord]]:
        to_verify: List[ClaimRecord] = []
        to_repair: List[ClaimRecord] = []
        for c in claims:
            if c.state in _TERMINAL:
                continue
            a = assessments.get(c.claim_id)
            if a is None:
                if evidence_store.get(c.claim_id) and bt.can_afford(
                    ControllerAction.VERIFY, c.claim_id, estimated_model_calls=1
                )[0]:
                    to_verify.append(c)
            elif (
                self.config.enable_repair
                and a.label in (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED)
                and c.claim_id not in repairs
                and bt.can_afford(ControllerAction.REPAIR, c.claim_id, estimated_model_calls=1)[0]
            ):
                to_repair.append(c)
        return to_verify, to_repair


class VerifyAllPolicy(_EligibilityBaseline):
    """Breadth-first: verify all claims in arrival order, then repair with leftover budget."""

    def select_next_action(self, claims, evidence_store, assessments, repairs, budget_tracker):
        if budget_tracker.is_exhausted():
            return ControllerAction.STOP, None, "Global verification budget is exhausted."
        to_verify, to_repair = self._eligible(claims, evidence_store, assessments, repairs, budget_tracker)
        if to_verify:
            c = to_verify[0]
            return ControllerAction.VERIFY, c, f"VerifyAll: verifying '{c.claim_id}' (arrival order)."
        if to_repair:
            c = to_repair[0]
            return ControllerAction.REPAIR, c, f"VerifyAll: all claims verified; repairing '{c.claim_id}'."
        return ControllerAction.STOP, None, "VerifyAll: nothing eligible under remaining budget."


class RandomPolicy(_EligibilityBaseline):
    """Uniform random choice over eligible (claim, action) pairs; seeded for reproducibility."""

    def __init__(self, config: Optional[ControllerPolicyConfig] = None, seed: int = 0) -> None:
        super().__init__(config)
        self.seed = seed
        self._rng = random.Random(seed)

    def select_next_action(self, claims, evidence_store, assessments, repairs, budget_tracker):
        if budget_tracker.is_exhausted():
            return ControllerAction.STOP, None, "Global verification budget is exhausted."
        to_verify, to_repair = self._eligible(claims, evidence_store, assessments, repairs, budget_tracker)
        pool = [(c, ControllerAction.VERIFY) for c in to_verify] + [(c, ControllerAction.REPAIR) for c in to_repair]
        if not pool:
            return ControllerAction.STOP, None, "Random: nothing eligible under remaining budget."
        c, action = self._rng.choice(pool)
        return action, c, f"Random(seed={self.seed}): picked {action.value} on '{c.claim_id}' from {len(pool)} options."
