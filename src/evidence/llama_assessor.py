"""Local Llama server evidence assessor adapter.

Connects to a llama.cpp OpenAI-compatible server to assess claim-evidence
alignment using natural language inference prompts.

DESIGN NOTES:
1. This adapter uses the same OpenAI-compatible API interface as the baseline
   vanilla RAG pipeline (http://127.0.0.1:8080/v1).
2. The assessor constructs a structured NLI-style prompt and parses the
   response into an EvidenceAssessment.
3. The adapter is designed to be testable WITHOUT a running server:
   - The constructor accepts an optional pre-configured OpenAI client.
   - Unit tests inject a mock client rather than connecting to the server.
4. The 'confidence' field in the output is the model's self-reported estimate,
   NOT a calibrated probability. This is explicitly documented in the output.
5. Temperature is set to 0.0 for deterministic assessment.
"""

from __future__ import annotations

import os

import json
import re
import time
from typing import Any, Dict, List, Optional

from src.claims.models import ClaimRecord
from .models import (
    AssessmentLabel,
    EvidenceAssessment,
    EvidencePassage,
    VerificationCost,
)
from .assessor import BaseAssessor


# Assessment prompt template
ASSESSMENT_PROMPT = """You are an evidence verification assistant. Your task is to determine whether the provided evidence supports, contradicts, or is insufficient to assess the given claim.

CLAIM:
{claim_text}

EVIDENCE:
{evidence_block}

Analyze ALL evidence passages JOINTLY and respond with a JSON object containing exactly these fields:
- "label": One of "SUPPORTED", "CONTRADICTED", "INSUFFICIENT", or "CONFLICTED"
- "confidence": A number between 0.0 and 1.0 representing your confidence in the label (this is your estimate, not a calibrated probability)
- "rationale": A concise explanation (1-3 sentences) of why you chose this label, mentioning which passage(s) contributed to your conclusion
- "supporting_spans": A list of objects, each with keys "passage_id" (the ID from the passage header) and "text" (exact excerpt from that passage supporting your conclusion)
- "limitations": Any caveats, ambiguity, or missing information that affects your assessment

RULES:
1. A high retrieval score does NOT mean the evidence supports the claim. You must evaluate the actual content.
2. Pay close attention to negations, dates, numbers, and temporal orderings.
3. If the evidence is irrelevant or does not address the claim specifically, use "INSUFFICIENT".
4. If different evidence passages support contradictory conclusions, use "CONFLICTED".
5. If you cannot decide reliably, prefer "INSUFFICIENT" over forcing a verdict.
6. Only use "SUPPORTED" when the evidence directly and specifically entails the claim.
7. Only use "CONTRADICTED" when the evidence directly and specifically negates the claim.
8. IMPORTANT: You may reason across multiple passages jointly. Two passages can together support a claim even if neither alone is sufficient. For example, if Passage A establishes that X is in location Y, and Passage B establishes that location Y is in State Z, then together they support a claim that X is in State Z.
9. In your rationale, explicitly state which passage(s) provided the key facts and how they combine to support or refute the claim.

Respond with ONLY the JSON object, no other text."""


def _format_evidence_block(
    evidence: List[EvidencePassage],
    max_passages: int = 4,
    max_passage_chars: int = 900,
) -> tuple:
    """Format evidence passages into a numbered block for the prompt.

    Enforces budget constraints:
    - Retains at most max_passages (prioritizing highest retrieval_score if available).
    - Truncates individual passage texts to max_passage_chars.
    - Records any truncation or omission in metadata.

    Returns:
        tuple of (formatted_text: str, was_truncated: bool, truncation_note: Optional[str])
    """
    if not evidence:
        return "(No evidence passages provided)", False, None

    omitted_count = 0
    truncated_passages = 0
    selected_evidence = list(evidence)

    if len(selected_evidence) > max_passages:
        omitted_count = len(selected_evidence) - max_passages
        selected_evidence.sort(
            key=lambda p: (p.retrieval_score is not None, p.retrieval_score or 0.0),
            reverse=True,
        )
        selected_evidence = selected_evidence[:max_passages]

    parts = []
    for i, passage in enumerate(selected_evidence):
        header = f"[Passage {i + 1}] (ID: {passage.evidence_id})"
        if passage.doc_title:
            header += f" Title: {passage.doc_title}"
        if passage.retrieval_score is not None:
            header += f" (retrieval_score={passage.retrieval_score:.4f})"

        text = passage.source_text
        if len(text) > max_passage_chars:
            text = text[:max_passage_chars] + "... [TRUNCATED]"
            truncated_passages += 1
        parts.append(f"{header}\n{text}")

    was_truncated = (omitted_count > 0) or (truncated_passages > 0)
    note = None
    if was_truncated:
        notes = []
        if omitted_count > 0:
            notes.append(f"Omitted {omitted_count} lower-ranked passages (budget: {max_passages})")
        if truncated_passages > 0:
            notes.append(f"Truncated {truncated_passages} passage(s) to {max_passage_chars} chars")
        note = "; ".join(notes)

    return "\n\n".join(parts), was_truncated, note


def _parse_assessment_response(
    raw_response: str,
    claim_id: str,
    evidence_ids: List[str],
) -> tuple:
    """Parse the LLM's JSON response into assessment fields.

    Returns:
        Tuple of (label, confidence, rationale, supporting_spans, limitations).
        Falls back to INSUFFICIENT with low confidence on parse failure.
    """
    # Strip markdown code fences if present
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    cleaned = cleaned.strip()

    data = None
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # Attempt recovery 1: model omitted trailing closing brace
        if cleaned.startswith("{") and not cleaned.endswith("}"):
            try:
                data = json.loads(cleaned + "\n}")
            except (json.JSONDecodeError, ValueError):
                pass
        # Attempt recovery 2: extract substring between first '{' and last '}'
        if data is None and "{" in cleaned:
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            if last_brace > first_brace:
                try:
                    data = json.loads(cleaned[first_brace:last_brace + 1])
                except (json.JSONDecodeError, ValueError):
                    pass
        # Attempt recovery 3: regex field extraction for valid LLM judgments
        if data is None:
            label_m = re.search(r'"label"\s*:\s*"([A-Za-z]+)"', cleaned)
            conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', cleaned)
            rat_m = re.search(r'"rationale"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', cleaned)
            if label_m:
                data = {
                    "label": label_m.group(1),
                    "confidence": float(conf_m.group(1)) if conf_m else 0.5,
                    "rationale": rat_m.group(1) if rat_m else "Extracted from partial response.",
                    "supporting_spans": [],
                    "limitations": None,
                }

    if data is None:
        return (
            AssessmentLabel.INSUFFICIENT,
            0.2,
            f"Failed to parse assessor response as JSON. Raw response: {raw_response[:200]}",
            {},
            "Assessment response could not be parsed. Result is unreliable.",
        )

    # Parse label
    raw_label = data.get("label", "INSUFFICIENT").upper().strip()
    try:
        label = AssessmentLabel(raw_label)
    except ValueError:
        label = AssessmentLabel.INSUFFICIENT

    # Parse confidence (clamp to [0.0, 1.0])
    try:
        confidence = float(data.get("confidence", 0.3))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.3

    rationale = str(data.get("rationale", "No rationale provided."))
    limitations = data.get("limitations")
    if limitations:
        limitations = str(limitations)

    # Parse supporting spans — map per passage ID when the model provides structured spans
    raw_spans = data.get("supporting_spans", [])
    supporting_spans: Dict[str, List[str]] = {}
    if isinstance(raw_spans, list) and raw_spans:
        for item in raw_spans:
            if isinstance(item, dict):
                # Structured format: {"passage_id": "...", "text": "..."}
                pid = str(item.get("passage_id", "")).strip()
                text = str(item.get("text", "")).strip()
                if text:
                    # Match against known evidence IDs (case-insensitive partial match)
                    matched_id = None
                    for eid in evidence_ids:
                        if pid and (pid == eid or pid in eid or eid in pid):
                            matched_id = eid
                            break
                    target_id = matched_id or (evidence_ids[0] if evidence_ids else "unknown")
                    supporting_spans.setdefault(target_id, []).append(text)
            elif isinstance(item, str) and item.strip():
                # Flat string format — assign to first evidence ID as fallback
                if evidence_ids:
                    supporting_spans.setdefault(evidence_ids[0], []).append(item.strip())

    return label, confidence, rationale, supporting_spans, limitations


class LlamaAssessor(BaseAssessor):
    """Evidence assessor using a local llama.cpp OpenAI-compatible server.

    Constructs NLI-style prompts and parses structured JSON responses.
    Designed to be testable with a mock OpenAI client.

    Args:
        client: An OpenAI-compatible client instance. If None, creates one
               connecting to the default local server URL.
        base_url: Server URL (default: http://127.0.0.1:8080/v1).
        model_name: Model identifier for the API. If None, auto-detects
                   from the server's model list.
        temperature: Generation temperature (default: 0.0 for determinism).
        max_tokens: Maximum response tokens (default: 512).
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        base_url: str = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1"),
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_passages: int = 4,
        max_passage_chars: int = 900,
    ) -> None:
        self._client = client
        self._base_url = base_url
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_passages = max_passages
        self._max_passage_chars = max_passage_chars
        self._initialized = client is not None

    def _ensure_client(self) -> None:
        """Lazily initialize the OpenAI client and discover the model."""
        if self._initialized:
            return

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for LlamaAssessor. "
                "Install it with: pip install openai"
            )

        self._client = OpenAI(base_url=self._base_url, api_key="local")

        if self._model_name is None:
            models = self._client.models.list()
            self._model_name = models.data[0].id

        self._initialized = True

    @property
    def assessor_id(self) -> str:
        model_suffix = self._model_name or "unknown"
        return f"llama_assessor_{model_suffix}"

    def assess(
        self,
        claim: ClaimRecord,
        evidence: List[EvidencePassage],
    ) -> EvidenceAssessment:
        """Assess claim-evidence alignment using the local Llama server."""
        self._ensure_client()

        claim_text = claim.normalized_text or claim.original_text
        evidence_ids = [e.evidence_id for e in evidence]

        # Handle empty evidence without LLM call
        if not evidence:
            return EvidenceAssessment(
                claim_id=claim.claim_id,
                evidence_ids=[],
                supporting_spans={},
                label=AssessmentLabel.INSUFFICIENT,
                confidence=0.95,
                rationale="No evidence passages provided for assessment.",
                limitations="Cannot assess without evidence.",
                cost=VerificationCost(model_calls=0),
                assessor_id=self.assessor_id,
            )

        # Build prompt with safe context budgeting
        evidence_block, was_truncated, truncation_note = _format_evidence_block(
            evidence,
            max_passages=self._max_passages,
            max_passage_chars=self._max_passage_chars,
        )
        prompt = ASSESSMENT_PROMPT.format(
            claim_text=claim_text,
            evidence_block=evidence_block,
        )

        # Call the model
        start_time = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as e:
            elapsed = time.monotonic() - start_time
            return EvidenceAssessment(
                claim_id=claim.claim_id,
                evidence_ids=evidence_ids,
                supporting_spans={},
                label=AssessmentLabel.INSUFFICIENT,
                confidence=0.0,
                rationale=f"LLM call failed: {type(e).__name__}: {str(e)[:200]}",
                limitations="Assessment could not be completed due to technical/model error.",
                cost=VerificationCost(
                    model_calls=1,
                    latency_seconds=elapsed,
                    model_id=self._model_name,
                ),
                assessor_id=self.assessor_id,
                is_error=True,
                error_type=type(e).__name__,
            )

        elapsed = time.monotonic() - start_time
        raw_content = response.choices[0].message.content or ""

        # Extract token usage if available
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None

        cost = VerificationCost(
            model_calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=elapsed,
            model_id=self._model_name,
        )

        # Parse response
        label, confidence, rationale, supporting_spans, limitations = _parse_assessment_response(
            raw_content, claim.claim_id, evidence_ids
        )

        # Record technical error if JSON parsing failed
        is_error = False
        error_type = None
        if rationale.startswith("Failed to parse assessor response as JSON"):
            is_error = True
            error_type = "JSONDecodeError"

        # Record truncation limitations if applicable
        if was_truncated and truncation_note:
            limitations = f"{limitations}; {truncation_note}" if limitations else truncation_note

        return EvidenceAssessment(
            claim_id=claim.claim_id,
            evidence_ids=evidence_ids,
            supporting_spans=supporting_spans,
            label=label,
            confidence=confidence,
            rationale=rationale,
            limitations=limitations,
            cost=cost,
            assessor_id=self.assessor_id,
            is_error=is_error,
            error_type=error_type,
        )
