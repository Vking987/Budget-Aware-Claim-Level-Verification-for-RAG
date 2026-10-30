"""Budget accounting and resource tracking module for Stage B controller.

Enforces separate global and per-claim budgets across distinct dimensions:
- Action step counts
- Model inference calls
- Input, output, and total token counts
- Elapsed wall-clock latency (seconds)

CRITICAL INVARIANT:
Different resource units are tracked separately. They must NEVER be combined
into a blended scalar.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.evidence.models import VerificationCost
from .models import ControllerAction, PerClaimBudget, VerificationBudget


class BudgetTracker:
    """Manages and audits resource consumption during verification."""

    def __init__(
        self,
        global_budget: Optional[VerificationBudget] = None,
        per_claim_budget: Optional[PerClaimBudget] = None,
    ) -> None:
        self.global_budget = global_budget or VerificationBudget()
        self.per_claim_budget = per_claim_budget or PerClaimBudget()

        # Cumulative global usage
        self.steps_taken: int = 0
        self.model_calls_used: int = 0
        self.input_tokens_used: int = 0
        self.output_tokens_used: int = 0
        self.total_tokens_used: int = 0
        self.latency_seconds_used: float = 0.0
        self.retrieval_operations: int = 0
        self.duplicate_passages_retrieved: int = 0
        self.unique_passages_retrieved: int = 0

        # Per-claim last-retrieval unique count (resets on each retrieval for that claim)
        self._last_retrieval_unique: Dict[str, int] = {}

        # Per-claim last-retrieval query string (used for stall detection)
        self._last_retrieval_query: Dict[str, Optional[str]] = {}

        # Per-claim usage: claim_id -> usage dict
        self._claim_usage: Dict[str, Dict[str, Any]] = {}

    def _ensure_claim_tracker(self, claim_id: str) -> Dict[str, Any]:
        if claim_id not in self._claim_usage:
            self._claim_usage[claim_id] = {
                "retrievals": 0,
                "verifications": 0,
                "repairs": 0,
                "model_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_seconds": 0.0,
                "duplicate_passages": 0,
                "unique_passages": 0,
            }
        return self._claim_usage[claim_id]

    def is_exhausted(self) -> bool:
        """Check whether any global hard budget limit has been reached."""
        gb = self.global_budget
        if gb.max_steps is not None and self.steps_taken >= gb.max_steps:
            return True
        if gb.max_model_calls is not None and self.model_calls_used >= gb.max_model_calls:
            return True
        if gb.max_input_tokens is not None and self.input_tokens_used >= gb.max_input_tokens:
            return True
        if gb.max_output_tokens is not None and self.output_tokens_used >= gb.max_output_tokens:
            return True
        if gb.max_total_tokens is not None and self.total_tokens_used >= gb.max_total_tokens:
            return True
        if gb.max_latency_seconds is not None and self.latency_seconds_used >= gb.max_latency_seconds:
            return True
        return False

    def can_afford(
        self,
        action: ControllerAction,
        claim_id: Optional[str] = None,
        estimated_model_calls: int = 0,
    ) -> Tuple[bool, str]:
        """Check if an action is eligible under current global and per-claim limits.

        Args:
            action: Proposed action.
            claim_id: Target claim ID if action is claim-specific.
            estimated_model_calls: Conservative reservation estimate for the action.

        Returns:
            Tuple of (can_afford_bool, rationale_message).
        """
        if action == ControllerAction.STOP or action == ControllerAction.DEFER:
            return True, "No budget consumed by control action"

        gb = self.global_budget
        # Check global steps
        if gb.max_steps is not None and self.steps_taken >= gb.max_steps:
            return False, f"Global max steps reached ({self.steps_taken}/{gb.max_steps})"

        # Check global model calls
        if gb.max_model_calls is not None:
            projected = self.model_calls_used + estimated_model_calls
            if projected > gb.max_model_calls:
                return False, (
                    f"Insufficient model call budget: {self.model_calls_used} used, "
                    f"+{estimated_model_calls} estimated > {gb.max_model_calls} max"
                )

        # Check global latency if set
        if gb.max_latency_seconds is not None and self.latency_seconds_used >= gb.max_latency_seconds:
            return False, f"Global latency budget reached ({self.latency_seconds_used:.2f}s)"

        # Check global tokens if set
        if gb.max_total_tokens is not None and self.total_tokens_used >= gb.max_total_tokens:
            return False, f"Global total tokens limit reached ({self.total_tokens_used})"

        # Check per-claim limits
        if claim_id is not None:
            pc = self.per_claim_budget
            usage = self._ensure_claim_tracker(claim_id)

            if action == ControllerAction.RETRIEVE:
                if usage["retrievals"] >= pc.max_retrievals:
                    return False, f"Claim '{claim_id}' reached max retrievals ({usage['retrievals']}/{pc.max_retrievals})"

            elif action == ControllerAction.VERIFY:
                if usage["verifications"] >= pc.max_verifications:
                    return False, f"Claim '{claim_id}' reached max verifications ({usage['verifications']}/{pc.max_verifications})"

            elif action == ControllerAction.REPAIR:
                if usage["repairs"] >= pc.max_repairs:
                    return False, f"Claim '{claim_id}' reached max repairs ({usage['repairs']}/{pc.max_repairs})"

            if pc.max_model_calls is not None:
                if usage["model_calls"] + estimated_model_calls > pc.max_model_calls:
                    return False, f"Claim '{claim_id}' would exceed max model calls ({usage['model_calls']}/{pc.max_model_calls})"

            if pc.max_total_tokens is not None and usage["total_tokens"] >= pc.max_total_tokens:
                return False, f"Claim '{claim_id}' reached token limit ({usage['total_tokens']}/{pc.max_total_tokens})"

        return True, "Eligible within budget limits"

    def record_action(
        self,
        action: ControllerAction,
        claim_id: Optional[str] = None,
        cost: Optional[VerificationCost] = None,
    ) -> None:
        """Account for actual resource consumption after action execution.

        Args:
            action: Action executed.
            claim_id: Target claim ID if applicable.
            cost: Actual cost returned by the executor.
        """
        self.steps_taken += 1

        # Track per-claim action count
        if claim_id is not None:
            usage = self._ensure_claim_tracker(claim_id)
            if action == ControllerAction.RETRIEVE:
                usage["retrievals"] += 1
            elif action == ControllerAction.VERIFY:
                usage["verifications"] += 1
            elif action == ControllerAction.REPAIR:
                usage["repairs"] += 1

        # Reconcile actual costs
        if cost is not None:
            calls = cost.model_calls or 0
            self.model_calls_used += calls

            p_tokens = cost.prompt_tokens or 0
            c_tokens = cost.completion_tokens or 0
            t_tokens = cost.total_tokens or (p_tokens + c_tokens)
            lat = cost.latency_seconds or 0.0

            self.input_tokens_used += p_tokens
            self.output_tokens_used += c_tokens
            self.total_tokens_used += t_tokens
            self.latency_seconds_used += lat

            if claim_id is not None:
                usage = self._ensure_claim_tracker(claim_id)
                usage["model_calls"] += calls
                usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + p_tokens
                usage["completion_tokens"] = usage.get("completion_tokens", 0) + c_tokens
                usage["input_tokens"] = usage["prompt_tokens"]
                usage["output_tokens"] = usage["completion_tokens"]
                usage["total_tokens"] += t_tokens
                usage["latency_seconds"] += lat

    def record_retrieval_passages(
        self,
        claim_id: str,
        unique_count: int,
        duplicate_count: int,
        query: Optional[str] = None,
    ) -> None:
        """Record unique vs duplicate retrieved passages for auditable retrieval efficiency.

        Args:
            claim_id: The claim that was retrieved for.
            unique_count: Number of new unique passages returned.
            duplicate_count: Number of duplicate passages returned.
            query: The query string used for this retrieval (enables query-aware stall detection).
        """
        self.retrieval_operations += 1
        self.unique_passages_retrieved += unique_count
        self.duplicate_passages_retrieved += duplicate_count

        # Track the unique count from the MOST RECENT retrieval for this claim (not cumulative)
        self._last_retrieval_unique[claim_id] = unique_count

        # Track the query used for the most recent retrieval (for stall detection)
        self._last_retrieval_query[claim_id] = query

        usage = self._ensure_claim_tracker(claim_id)
        usage["unique_passages"] = usage.get("unique_passages", 0) + unique_count
        usage["duplicate_passages"] = usage.get("duplicate_passages", 0) + duplicate_count

    def last_retrieval_unique_count(self, claim_id: str) -> int:
        """Return the number of unique passages yielded by the LAST retrieval for this claim.

        Returns 0 if no retrieval has been recorded yet for this claim.
        """
        return self._last_retrieval_unique.get(claim_id, 0)

    def is_retrieval_stalled(
        self,
        claim_id: str,
        current_query: Optional[str] = None,
    ) -> bool:
        """Return True only when retrieval is demonstrably stalled for this claim.

        Stalled is defined as:
        - A prior retrieval was recorded AND
        - That retrieval returned zero unique passages AND
        - The current query is identical to the prior retrieval query (or both are None).

        If the current query differs from the prior query, the retrieval is NOT stalled
        because a different query may return different results.

        Args:
            claim_id: The claim to check.
            current_query: The query string about to be used for retrieval.

        Returns:
            True if identical retry would be pointless; False if a new query or no prior attempt.
        """
        last_unique = self._last_retrieval_unique.get(claim_id, None)
        if last_unique is None:
            # No prior retrieval recorded — not stalled
            return False
        if last_unique > 0:
            # Prior retrieval found new passages — not stalled
            return False
        # Prior retrieval returned 0 unique passages.  Stalled only if same query.
        last_query = self._last_retrieval_query.get(claim_id)
        if current_query is not None and last_query is not None and current_query != last_query:
            # Query changed — allow a fresh attempt
            return False
        return True

    def get_remaining_budget(self) -> Dict[str, Any]:
        """Compute remaining budget in distinct resource units."""
        gb = self.global_budget
        return {
            "remaining_steps": max(0, gb.max_steps - self.steps_taken) if gb.max_steps is not None else None,
            "remaining_model_calls": max(0, gb.max_model_calls - self.model_calls_used) if gb.max_model_calls is not None else None,
            "remaining_prompt_tokens": max(0, gb.max_input_tokens - self.input_tokens_used) if gb.max_input_tokens is not None else None,
            "remaining_completion_tokens": max(0, gb.max_output_tokens - self.output_tokens_used) if gb.max_output_tokens is not None else None,
            "remaining_input_tokens": max(0, gb.max_input_tokens - self.input_tokens_used) if gb.max_input_tokens is not None else None,
            "remaining_output_tokens": max(0, gb.max_output_tokens - self.output_tokens_used) if gb.max_output_tokens is not None else None,
            "remaining_total_tokens": max(0, gb.max_total_tokens - self.total_tokens_used) if gb.max_total_tokens is not None else None,
            "remaining_latency_seconds": max(0.0, gb.max_latency_seconds - self.latency_seconds_used) if gb.max_latency_seconds is not None else None,
        }

    def get_global_cost(self) -> Dict[str, Any]:
        """Get snapshot of cumulative global resource consumption."""
        return {
            "steps_taken": self.steps_taken,
            "model_calls": self.model_calls_used,
            "prompt_tokens": self.input_tokens_used,
            "completion_tokens": self.output_tokens_used,
            "input_tokens": self.input_tokens_used,
            "output_tokens": self.output_tokens_used,
            "total_tokens": self.total_tokens_used,
            "latency_seconds": round(self.latency_seconds_used, 4),
            "retrieval_operations": self.retrieval_operations,
            "unique_passages_retrieved": self.unique_passages_retrieved,
            "duplicate_passages_retrieved": self.duplicate_passages_retrieved,
        }

    def get_claim_cost(self, claim_id: str) -> Dict[str, Any]:
        """Get snapshot of resource consumption for a specific claim."""
        usage = self._claim_usage.get(claim_id, {
            "retrievals": 0,
            "verifications": 0,
            "repairs": 0,
            "model_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "unique_passages": 0,
            "duplicate_passages": 0,
        })
        return {
            "retrievals": usage["retrievals"],
            "verifications": usage["verifications"],
            "repairs": usage["repairs"],
            "model_calls": usage["model_calls"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage["total_tokens"],
            "latency_seconds": round(usage["latency_seconds"], 4),
            "unique_passages": usage.get("unique_passages", 0),
            "duplicate_passages": usage.get("duplicate_passages", 0),
        }
