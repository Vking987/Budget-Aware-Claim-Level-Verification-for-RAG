"""Replaceable interfaces and mock implementations for Stage B actions.

Defines abstract contracts and offline deterministic mocks for:
- Evidence Retrieval (RETRIEVE)
- Claim Repair / Qualification (REPAIR)

Reuses BaseAssessor from src/evidence/assessor.py for VERIFY.
"""

from __future__ import annotations

import os

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from src.claims.models import ClaimRecord
from src.evidence.models import EvidenceAssessment, EvidencePassage, VerificationCost
from .models import RepairResult


# ---------------------------------------------------------------------------
# Repair prompt template used by LLMRepairer
# ---------------------------------------------------------------------------
_REPAIR_PROMPT = """\
You are a fact-checking assistant. A claim was assessed as {assessment_label} by evidence.

ORIGINAL CLAIM:
{claim_text}

ASSESSMENT RATIONALE:
{rationale}

SUPPORTING EVIDENCE PASSAGES:
{evidence_block}

Task: Produce a minimally revised version of the claim that is directly supported by the evidence above.
- If the claim is contradicted, correct it to match the evidence.
- If the evidence is insufficient or conflicting, qualify the claim explicitly (e.g., "According to [source], ...").
- Do NOT invent facts absent from the evidence.
- Do NOT change the topic or scope of the claim.

Respond with ONLY a JSON object containing exactly these fields:
- "revised_claim": The revised claim text (string, one sentence).
- "repair_reason": A brief explanation of what changed and why (string, 1-2 sentences).
- "confidence": Your confidence that the revision is accurate (float, 0.0-1.0).

JSON only, no other text."""


class BaseRetriever(ABC):
    """Abstract interface for evidence retrieval during verification."""

    @abstractmethod
    def retrieve(
        self,
        claim: ClaimRecord,
        top_k: int = 3,
    ) -> Tuple[List[EvidencePassage], VerificationCost]:
        """Fetch candidate evidence passages for a given claim.

        Args:
            claim: Target claim record.
            top_k: Number of passages to return.

        Returns:
            Tuple of (retrieved_passages, verification_cost).
        """
        ...


class MockRetriever(BaseRetriever):
    """Deterministic, offline mock retriever for testing.

    Allows tests to pre-program passages per claim ID or fall back to keyword
    synthesis. Does NOT require Qdrant or an embedding model.
    """

    def __init__(
        self,
        canned_evidence: Optional[Dict[str, List[EvidencePassage]]] = None,
        default_cost: Optional[VerificationCost] = None,
    ) -> None:
        self.canned_evidence = canned_evidence or {}
        self.default_cost = default_cost or VerificationCost(model_calls=0, latency_seconds=0.001)
        self.call_history: List[str] = []

    def set_evidence_for_claim(self, claim_id: str, passages: List[EvidencePassage]) -> None:
        self.canned_evidence[claim_id] = passages

    def retrieve(
        self,
        claim: ClaimRecord,
        top_k: int = 3,
    ) -> Tuple[List[EvidencePassage], VerificationCost]:
        start = time.monotonic()
        self.call_history.append(claim.claim_id)

        passages = self.canned_evidence.get(claim.claim_id, [])
        elapsed = time.monotonic() - start
        cost = VerificationCost(
            model_calls=self.default_cost.model_calls,
            prompt_tokens=self.default_cost.prompt_tokens,
            completion_tokens=self.default_cost.completion_tokens,
            total_tokens=self.default_cost.total_tokens,
            latency_seconds=elapsed,
            model_id="mock_retriever",
        )
        return passages[:top_k], cost


class BaseRepairer(ABC):
    """Abstract interface for claim qualification and correction."""

    @abstractmethod
    def repair(
        self,
        claim: ClaimRecord,
        assessment: EvidenceAssessment,
        evidence: List[EvidencePassage],
    ) -> Tuple[RepairResult, VerificationCost]:
        """Generate a structured repair or qualification for a contested claim.

        Args:
            claim: Target claim record.
            assessment: Contradicted or conflicted assessment.
            evidence: Available supporting/refuting evidence passages.

        Returns:
            Tuple of (RepairResult, VerificationCost).
        """
        ...


class MockRepairer(BaseRepairer):
    """Deterministic, offline mock repairer for testing.

    Generates explicit, non-destructive revisions with documented rationale
    and evidence IDs without requiring an LLM inference call.
    """

    def __init__(
        self,
        default_cost: Optional[VerificationCost] = None,
    ) -> None:
        self.default_cost = default_cost or VerificationCost(model_calls=0, latency_seconds=0.001)
        self.call_history: List[str] = []

    def repair(
        self,
        claim: ClaimRecord,
        assessment: EvidenceAssessment,
        evidence: List[EvidencePassage],
    ) -> Tuple[RepairResult, VerificationCost]:
        start = time.monotonic()
        self.call_history.append(claim.claim_id)

        evidence_ids = assessment.evidence_ids or [e.evidence_id for e in evidence]
        reason = (
            f"Claim qualification generated in response to {assessment.label.value} assessment. "
            f"Rationale: {assessment.rationale}"
        )

        # Non-destructive proposed revision
        proposed = f"[Qualified]: According to evidence ({', '.join(evidence_ids)}), {claim.original_text}"

        repair_res = RepairResult(
            claim_id=claim.claim_id,
            original_text=claim.original_text,
            proposed_revision=proposed,
            repair_reason=reason,
            supporting_evidence_ids=evidence_ids,
            confidence=0.70,
        )

        elapsed = time.monotonic() - start
        cost = VerificationCost(
            model_calls=self.default_cost.model_calls,
            prompt_tokens=self.default_cost.prompt_tokens,
            completion_tokens=self.default_cost.completion_tokens,
            total_tokens=self.default_cost.total_tokens,
            latency_seconds=elapsed,
            model_id="mock_repairer",
        )
        repair_res.cost = cost
        return repair_res, cost


class LLMRepairer(BaseRepairer):
    """Production repairer that calls the local llama.cpp OpenAI-compatible server.

    Constructs a structured fact-checking prompt from the claim, its assessment,
    and the available evidence passages. Parses the model's JSON response into a
    RepairResult. Falls back to MockRepairer behaviour on any exception so the
    controller loop is never broken by a server or parse error.

    Args:
        base_url: llama.cpp server URL (default: http://127.0.0.1:8080/v1).
        model_name: Model identifier. None = auto-detected from server.
        temperature: Generation temperature (default: 0.0 for determinism).
        max_tokens: Max completion tokens (default: 256).
        client: Pre-configured OpenAI-compatible client (for testing / injection).
    """

    def __init__(
        self,
        base_url: str = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1"),
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        client: Optional[Any] = None,
        max_evidence_passages: int = 3,
        max_passage_chars: int = 600,
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_evidence_passages = max_evidence_passages
        self.max_passage_chars = max_passage_chars
        self._client = client
        self._initialized = client is not None

    def _ensure_client(self) -> None:
        if self._initialized:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for LLMRepairer. "
                "Install with: pip install openai"
            ) from exc
        try:
            self._client = OpenAI(base_url=self.base_url, api_key="local", timeout=30.0)
            if self.model_name is None:
                models = self._client.models.list()
                self.model_name = models.data[0].id if models.data else "local_model"
            self._initialized = True
        except Exception as exc:
            raise RuntimeError(
                f"LLMRepairer failed to connect to LLM server at {self.base_url}: {exc}"
            ) from exc

    def _format_evidence(self, evidence: List[EvidencePassage]) -> str:
        selected = sorted(
            evidence,
            key=lambda p: (p.retrieval_score is not None, p.retrieval_score or 0.0),
            reverse=True,
        )[: self.max_evidence_passages]
        parts: List[str] = []
        for i, p in enumerate(selected):
            title = f" Title: {p.doc_title}" if p.doc_title else ""
            text = p.source_text[: self.max_passage_chars]
            if len(p.source_text) > self.max_passage_chars:
                text += "... [TRUNCATED]"
            parts.append(f"[{i + 1}] (ID: {p.evidence_id}){title}\n{text}")
        return "\n\n".join(parts) if parts else "(No evidence passages provided)"

    @staticmethod
    def _parse_repair_response(raw: str) -> Tuple[str, str, float]:
        """Return (revised_claim, repair_reason, confidence) from JSON response.

        Returns empty revision and descriptive reason if parsing fails.
        """
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        data: Optional[Dict[str, Any]] = None
        for attempt_str in [cleaned, cleaned + "\n}"]:
            try:
                data = json.loads(attempt_str)
                break
            except (json.JSONDecodeError, ValueError):
                pass

        if data is None and "{" in cleaned:
            first = cleaned.find("{")
            last = cleaned.rfind("}")
            if last > first:
                try:
                    data = json.loads(cleaned[first : last + 1])
                except (json.JSONDecodeError, ValueError):
                    pass

        if data is None:
            return "", f"Parse failure. Raw: {raw[:200]}", 0.0

        revised = str(data.get("revised_claim", "")).strip()
        reason = str(data.get("repair_reason", "No reason provided.")).strip()
        try:
            conf = float(data.get("confidence", 0.5))
            conf = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            conf = 0.5
        return revised, reason, conf

    def repair(
        self,
        claim: ClaimRecord,
        assessment: EvidenceAssessment,
        evidence: List[EvidencePassage],
    ) -> Tuple[RepairResult, VerificationCost]:
        """Generate an LLM-grounded revision for a contradicted/conflicted claim.

        In live mode, does NOT silently fall back to MockRepairer.
        Raises RuntimeError on connection or server error so the failure
        is surfaced explicitly to the controller/pipeline trace.
        """
        start = time.monotonic()
        self._ensure_client()

        evidence_block = self._format_evidence(evidence)
        assessment_label = assessment.label.value if assessment else "UNKNOWN"
        rationale = assessment.rationale if assessment else "No rationale available."

        prompt = _REPAIR_PROMPT.format(
            assessment_label=assessment_label,
            claim_text=claim.normalized_text or claim.original_text,
            rationale=rationale,
            evidence_block=evidence_block,
        )

        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw_content = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            model_calls = 1
        except Exception as exc:
            raise RuntimeError(
                f"LLMRepairer call failed ({type(exc).__name__}): {exc}"
            ) from exc

        # Parse the structured response
        revised, reason, conf = self._parse_repair_response(raw_content)

        evidence_ids = [e.evidence_id for e in evidence]

        # Do not fabricate a repaired claim when the LLM produces no revision or parse fails
        if not revised:
            revised = ""
            reason = f"LLM repair generation failed or was unparseable. {reason}".strip()
            conf = 0.0

        elapsed = time.monotonic() - start
        cost = VerificationCost(
            model_calls=model_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_seconds=elapsed,
            model_id=self.model_name or "llm_repairer",
        )

        repair_res = RepairResult(
            claim_id=claim.claim_id,
            original_text=claim.original_text,
            proposed_revision=revised,
            repair_reason=reason,
            supporting_evidence_ids=evidence_ids,
            confidence=conf,
            cost=cost,
        )
        return repair_res, cost
