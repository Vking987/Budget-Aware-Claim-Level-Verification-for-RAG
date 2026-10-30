"""Final Answer Synthesis Module for Stage E / Milestone Completion.

Synthesizes an evidence-grounded final answer from verified claims,
excluding refuted/unresolved claims and preserving citations and uncertainty.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from src.claims.models import ClaimRecord, ClaimState
from src.controller.models import ControllerTrace
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost
from .models import SynthesisResult


SYNTHESIS_SYSTEM_PROMPT = """You rewrite a list of verified claims as a short answer.

RULES:
1. State only the verified claims you are given. Add no other fact, number or name, and never use outside knowledge.
2. Keep every verified claim; do not repeat any claim and do not invent connecting facts.
3. Cite with the exact source title given for the claim, inside square brackets, at the end of its line. Use square brackets for nothing else.
4. Your output is checked automatically; if it adds or drops a claim, repeats a sentence, or cites a source that does not support its line, it is discarded and replaced by a deterministic answer."""


def _trim_to_last_sentence(text: str) -> str:
    """Trim text to end at the last complete sentence boundary.

    Used to salvage a clean sentence from a response that was cut off mid-sentence
    by a token limit (finish_reason == 'length').  Returns the trimmed text, or
    the original text unchanged if no sentence boundary is found (caller should
    then fall back to the full deterministic fallback path).

    A 'sentence boundary' is the position of the last '.', '!', or '?' in the
    text, ignoring trailing whitespace.  We do NOT use heuristics such as
    capitalisation of the following character because the text may be truncated
    before any following character exists.
    """
    stripped = text.rstrip()
    # Find rightmost sentence-ending punctuation
    last_boundary = max(
        stripped.rfind("."),
        stripped.rfind("!"),
        stripped.rfind("?"),
    )
    if last_boundary <= 0:
        # No boundary found — return unchanged so caller can fall back
        return text
    return stripped[: last_boundary + 1]




SYNTHESIS_USER_PROMPT_TEMPLATE = """USER QUESTION:
{question}

VERIFIED CLAIMS (the ONLY facts you may state, each with its source title):
{verified_claims_block}

INSTRUCTIONS:
Rewrite the verified claims above as a short answer. Use only the words and facts in the claims: add nothing, drop nothing, and do not repeat a claim. Put each claim's source title, copied exactly, inside square brackets at the end of its line. Use square brackets only for source titles. Do not mention anything that is not listed above."""


class BaseSynthesizer(ABC):
    """Abstract interface for final answer synthesis."""

    @abstractmethod
    def synthesize(
        self,
        question: str,
        initial_draft_answer: str,
        claims: List[ClaimRecord],
        evidence_store: Dict[str, List[EvidencePassage]],
        assessments: Optional[Dict[str, EvidenceAssessment]] = None,
        trace: Optional[ControllerTrace] = None,
    ) -> SynthesisResult:
        """Synthesize final grounded answer from verified claims and evidence."""
        ...


class LLMAnswerSynthesizer(BaseSynthesizer):
    """Evidence-grounded answer synthesizer using the local LLM endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 384,
        client: Optional[Any] = None,
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client
        self._initialized = client is not None

    def _ensure_client(self) -> None:
        if self._initialized:
            return
        from openai import OpenAI
        self._client = OpenAI(base_url=self.base_url, api_key="local", timeout=30.0)
        if self.model_name is None:
            models = self._client.models.list()
            self.model_name = models.data[0].id if models.data else "local_model"
        self._initialized = True

    def synthesize(
        self,
        question: str,
        initial_draft_answer: str,
        claims: List[ClaimRecord],
        evidence_store: Dict[str, List[EvidencePassage]],
        assessments: Optional[Dict[str, EvidenceAssessment]] = None,
        trace: Optional[ControllerTrace] = None,
    ) -> SynthesisResult:
        start_time = time.monotonic()

        # Categorize claims based on controller verification state
        verified_texts: List[str] = []
        unresolved_texts: List[str] = []
        contradicted_texts: List[str] = []
        uncertainty_notes: List[str] = []
        cited_sources: Set[str] = set()

        assmts = assessments or {}
        repaired_texts: List[str] = []
        repairs_by_claim: Dict[str, Any] = {}
        if trace:
            for step in trace.steps:
                if step.claim_id and step.assessment_label:
                    if step.assessment_label in ("CONTRADICTED", "CONFLICTED"):
                        contradicted_texts.append(step.claim_text or step.claim_id)
                if step.claim_id and step.repair_data:
                    rep_rev = step.repair_data.get("proposed_revision")
                    rv = step.repair_data.get("reverification") or {}
                    # Only a repair whose revised text was re-verified as SUPPORTED is
                    # treated as grounded; others stay unresolved/contradicted.
                    if rep_rev and rep_rev.strip() and rv.get("verified"):
                        repairs_by_claim[step.claim_id] = step.repair_data
                        if rep_rev not in repaired_texts:
                            repaired_texts.append(rep_rev)

        for c in claims:
            txt = c.normalized_text or c.original_text
            c_assmt = assmts.get(c.claim_id)

            if c.claim_id in repairs_by_claim:
                rep_rev = repairs_by_claim[c.claim_id].get("proposed_revision")
                sup_ids = set(repairs_by_claim[c.claim_id].get("supporting_evidence_ids", []))
                verified_texts.append(rep_rev)
                for p in evidence_store.get(c.claim_id, []):
                    if p.doc_title and (not sup_ids or p.evidence_id in sup_ids):
                        cited_sources.add(p.doc_title)
            elif c.state == ClaimState.VERIFIED:
                verified_texts.append(txt)
                # Collect sources from evidence store
                for p in evidence_store.get(c.claim_id, []):
                    if p.doc_title:
                        cited_sources.add(p.doc_title)
            elif c_assmt and c_assmt.label in (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED):
                contradicted_texts.append(txt)
                uncertainty_notes.append(f"Claim refuted or conflicted: '{txt}'")
            else:
                unresolved_texts.append(txt)
                uncertainty_notes.append(f"Unresolved claim (insufficient evidence or budget): '{txt}'")

        # Deduplicate
        verified_texts = list(dict.fromkeys(verified_texts))
        unresolved_texts = list(dict.fromkeys(unresolved_texts))
        contradicted_texts = list(dict.fromkeys(contradicted_texts))

        # Check if any verified evidence exists
        has_verified_evidence = len(verified_texts) > 0
        has_unresolved = len(unresolved_texts) > 0 or len(contradicted_texts) > 0

        # Collect ONLY passages linked to VERIFIED / REPAIRED claims for citations.
        # Prefer passages that have supporting spans recorded in evidence_provenance;
        # fall back to all passages for that verified claim.
        cited_sources = set()  # Reset — earlier loop included unverified claims, discard it
        verified_claim_ids = {c.claim_id for c in claims if (c.state == ClaimState.VERIFIED or c.claim_id in repairs_by_claim)}
        for c in claims:
            if c.claim_id in repairs_by_claim:
                sup_ids = set(repairs_by_claim[c.claim_id].get("supporting_evidence_ids", []))
                for p in evidence_store.get(c.claim_id, []):
                    if p.doc_title and (not sup_ids or p.evidence_id in sup_ids):
                        cited_sources.add(p.doc_title)
                continue
            if c.state != ClaimState.VERIFIED:
                continue
            claim_passages = evidence_store.get(c.claim_id, [])
            # Determine which passage IDs actually provided supporting spans
            provenance = c.evidence_provenance or {}
            supporting_spans = provenance.get("supporting_spans", {})
            if supporting_spans:
                # Only cite passages that have a supporting span entry
                supporting_ids = set(supporting_spans.keys())
                for p in claim_passages:
                    if p.evidence_id in supporting_ids and p.doc_title:
                        cited_sources.add(p.doc_title)
            else:
                # No span info: fall back to all passages for this verified claim
                for p in claim_passages:
                    if p.doc_title:
                        cited_sources.add(p.doc_title)

        # Claim-faithful grounding. The answer is built only from VERIFIED claims (and repairs that passed
        # re-verification). Each item carries the passage that literally supports it (overlap AND all numbers and
        # names present); passages the provenance logic above did not allow are never candidates, so this can only
        # narrow citations. Unresolved / contradicted claims and repairs that failed re-verification are collected
        # as `excluded_texts`: none of their distinctive content may reach the answer.
        from .grounding import build_items, stitch_answer, validate_answer
        items = build_items(claims, repairs_by_claim, evidence_store, verified_claim_ids, allowed_titles=cited_sources)
        excluded_texts = list(unresolved_texts) + list(contradicted_texts)
        if trace:
            for _step in trace.steps:
                _rev = ((_step.repair_data or {}).get("proposed_revision") or "").strip()
                if _rev and _step.claim_id not in repairs_by_claim:
                    excluded_texts.append(_rev)
        cited_sources = {i["title"] for i in items if i.get("title")}
        for _i in items:
            if not _i.get("title"):
                uncertainty_notes.append(
                    f"Verified claim has no retrieved passage containing its key facts {_i['missing_facts']}: '{_i['text']}'"
                )

        # Collect context passages for the LLM prompt — ONLY from verified claims.
        # Including passages from unresolved claims balloons the prompt and can cause
        # the model to ramble about irrelevant material, leading to truncated answers.
        all_passages: List[EvidencePassage] = []
        seen_p_ids: set = set()
        for cid, p_list in evidence_store.items():
            if cid not in verified_claim_ids:
                continue  # Exclude non-verified and synthetic evidence keys
            for p in p_list:
                if p.evidence_id not in seen_p_ids:
                    seen_p_ids.add(p.evidence_id)
                    all_passages.append(p)

        # Sort passages by retrieval score if present
        all_passages.sort(key=lambda p: (p.retrieval_score is not None, p.retrieval_score or 0.0), reverse=True)
        top_passages = all_passages[:4]  # Tighter bound: 4 passages avoids context bloat

        # The optional LLM rewrite sees ONLY the verified claims (with their source titles): no draft, no
        # unresolved claims, no raw passages, so it has nothing else to restate. Its output is validated below.
        verified_block = "\n".join(
            f"- {i['text']}" + (f" (source title: {i['title']})" if i.get("title") else "") for i in items
        ) if items else "(None)"

        # If zero verified claims, return safe abstention immediately — do NOT call LLM with the draft.
        # This prevents the model from echoing the draft as a factual answer.
        if not verified_texts:
            elapsed = time.monotonic() - start_time
            # Unresolved/contradicted claims are counted, never quoted: quoting them would restate unverified content.
            limitation_parts = []
            if unresolved_texts:
                limitation_parts.append(f"{len(unresolved_texts)} claim(s) could not be verified and are not included.")
            if contradicted_texts:
                limitation_parts.append(f"{len(contradicted_texts)} claim(s) were contradicted by the evidence and are not included.")
            limitation_detail = " ".join(limitation_parts) if limitation_parts else ""
            safe_answer = (
                "The available evidence is insufficient to verify the answer to this question. "
                + limitation_detail
            ).strip()
            return SynthesisResult(
                question=question,
                initial_draft_answer=initial_draft_answer,
                final_answer=safe_answer,
                verified_claims=[],
                unresolved_claims=unresolved_texts,
                contradicted_claims=contradicted_texts,
                cited_sources=[],
                uncertainty_notes=uncertainty_notes + ["No claims were verified against retrieved evidence."],
                is_grounded=False,
                has_unresolved_limitations=True,
                cost=VerificationCost(model_calls=0, latency_seconds=elapsed, model_id=self.model_name),
                synthesis_method="safe_abstention",
            )

        # If no evidence passages are available (should be rare given above), return direct limitation
        if not top_passages:
            elapsed = time.monotonic() - start_time
            return SynthesisResult(
                question=question,
                initial_draft_answer=initial_draft_answer,
                final_answer="The available evidence does not contain sufficient verified information to answer the question.",
                verified_claims=[],
                unresolved_claims=unresolved_texts,
                contradicted_claims=contradicted_texts,
                cited_sources=[],
                uncertainty_notes=["No evidence passages or verified claims available."],
                is_grounded=True,
                has_unresolved_limitations=True,
                cost=VerificationCost(model_calls=0, latency_seconds=elapsed, model_id=self.model_name),
                synthesis_method="fallback_limitation",
            )

        user_content = SYNTHESIS_USER_PROMPT_TEMPLATE.format(
            question=question,
            verified_claims_block=verified_block,
        )

        error_msg = None
        final_answer = ""
        prompt_tokens = 0
        completion_tokens = 0
        model_calls_used = 0
        # Minimum plausible answer length: if shorter than this, treat as truncated
        MIN_ANSWER_LENGTH = 20
        MAX_RETRIES = 1  # Retry at most once on truncation

        self._ensure_client()
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                candidate = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                model_calls_used += 1
                prompt_tokens += pt
                completion_tokens += ct

                # Detect truncated / empty response
                finish_reason = ""
                if resp.choices:
                    finish_reason = getattr(resp.choices[0], "finish_reason", "") or ""

                # On length truncation, attempt to salvage a complete sentence
                # before deciding to retry or fall back.
                if finish_reason == "length" and candidate.strip():
                    trimmed = _trim_to_last_sentence(candidate.strip())
                    if trimmed:
                        candidate = trimmed
                        # If trimming produced a complete sentence, treat as non-truncated
                        finish_reason = "stop"

                is_truncated = (
                    finish_reason == "length"
                    or (candidate.strip() and len(candidate.strip()) < MIN_ANSWER_LENGTH)
                )
                if not is_truncated or attempt == MAX_RETRIES:
                    final_answer = candidate
                    break
                # else: retry once
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {str(exc)}"
                break

        # Deterministic fallback if LLM failed or produced empty output: stitch the verified claims themselves.
        polish_status = "llm_accepted"
        if not final_answer.strip():
            if error_msg is None:
                error_msg = "Empty or truncated synthesis response after retry."
            final_answer = stitch_answer(items) if items else \
                "The available evidence is insufficient to verify the answer to this question."
            polish_status = "stitched_fallback"

        # Sanitize final_answer against placeholder citations emitted by LLM
        final_answer = final_answer.replace("[Document Title]", "").replace("[Title]", "")
        import re
        final_answer = re.sub(r'\[Passage\s+\d+\]', '', final_answer, flags=re.IGNORECASE)
        # Clean up any leftover double spaces
        final_answer = re.sub(r' +', ' ', final_answer).strip()

        # ---- Enforced grounding validation -------------------------------------------------------------
        # An LLM rewrite is accepted only if it passes validate_answer (whole verified claims only, no
        # unresolved-claim content, no repeats, nothing dropped, citations match their lines). Otherwise the
        # deterministic stitched answer replaces it. Whatever is returned is re-validated; if even that fails
        # (should be impossible) the answer is a safe abstention, never unvalidated text.
        answer_violations: List[str] = []
        if items:
            answer_violations = validate_answer(final_answer, items, excluded_texts)
            if polish_status == "llm_accepted" and answer_violations:
                uncertainty_notes.append(
                    f"LLM rewrite rejected by grounding validation ({len(answer_violations)} violation(s); first: "
                    f"{answer_violations[:2]}); the deterministic stitched answer was used instead."
                )
                final_answer = stitch_answer(items)
                polish_status = "llm_rejected_stitched"
                answer_violations = validate_answer(final_answer, items, excluded_texts)
            if answer_violations:
                error_msg = ((error_msg + "; ") if error_msg else "") + f"grounding validation failed: {answer_violations[:2]}"
                final_answer = "The available evidence is insufficient to verify the answer to this question."
                polish_status = "validation_failed_abstention"
        if has_unresolved:  # fixed, non-factual disclosure; appended after validation and never quotes a claim
            from .citations import LIMITATION_NOTE
            final_answer = final_answer.rstrip() + "\n" + LIMITATION_NOTE

        # Strict validation of cited_sources: exclude placeholders or empty strings
        clean_cited_sources = sorted([
            s for s in cited_sources 
            if s and s not in ("Document Title", "Title") and not re.match(r'^Passage\s+\d+$', s, flags=re.IGNORECASE)
        ])

        # Grounding requires BOTH verified/repaired statements AND valid, resolved cited sources
        is_grounded = bool(verified_texts) and len(clean_cited_sources) > 0

        elapsed = time.monotonic() - start_time
        cost = VerificationCost(
            model_calls=model_calls_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_seconds=elapsed,
            model_id=self.model_name,
        )

        return SynthesisResult(
            question=question,
            initial_draft_answer=initial_draft_answer,
            final_answer=final_answer.strip(),
            verified_claims=verified_texts,
            unresolved_claims=unresolved_texts,
            contradicted_claims=contradicted_texts,
            cited_sources=clean_cited_sources,
            uncertainty_notes=uncertainty_notes,
            is_grounded=is_grounded,
            has_unresolved_limitations=has_unresolved or (len(repaired_texts) > 0 and len(repaired_texts) == len(verified_texts)),
            cost=cost,
            synthesis_method={
                "llm_accepted": "llm_grounded",
                "llm_rejected_stitched": "deterministic_stitched",
                "stitched_fallback": "fallback_grounded",
                "validation_failed_abstention": "validation_failed_abstention",
            }[polish_status],
            error=error_msg,
            claim_citations=[{"claim_id": i["claim_id"], "claim": i["text"], "title": i.get("title"),
                              "evidence_ids": i["evidence_ids"], "overlap": i["overlap"],
                              "missing_facts": i["missing_facts"]} for i in items],
            answer_violations=answer_violations,
            polish_status=polish_status,
            grounding_enforced=True,
        )


class MockAnswerSynthesizer(BaseSynthesizer):
    """Deterministic offline answer synthesizer for tests and dry runs."""

    def __init__(self, default_cost: Optional[VerificationCost] = None) -> None:
        self.default_cost = default_cost or VerificationCost(model_calls=0, latency_seconds=0.001)

    def synthesize(
        self,
        question: str,
        initial_draft_answer: str,
        claims: List[ClaimRecord],
        evidence_store: Dict[str, List[EvidencePassage]],
        assessments: Optional[Dict[str, EvidenceAssessment]] = None,
        trace: Optional[ControllerTrace] = None,
    ) -> SynthesisResult:
        start_time = time.monotonic()
        assmts = assessments or {}

        verified_texts: List[str] = []
        unresolved_texts: List[str] = []
        contradicted_texts: List[str] = []
        uncertainty_notes: List[str] = []
        cited_sources: Set[str] = set()

        repaired_texts: List[str] = []
        repairs_by_claim: Dict[str, Any] = {}
        if trace:
            for step in trace.steps:
                if step.claim_id and step.assessment_label:
                    if step.assessment_label in ("CONTRADICTED", "CONFLICTED"):
                        c_txt = step.claim_text or step.claim_id
                        if c_txt not in contradicted_texts:
                            contradicted_texts.append(c_txt)
                if step.claim_id and step.repair_data:
                    rep_rev = step.repair_data.get("proposed_revision")
                    rv = step.repair_data.get("reverification") or {}
                    # Only a repair whose revised text was re-verified as SUPPORTED is
                    # treated as grounded; others stay unresolved/contradicted.
                    if rep_rev and rep_rev.strip() and rv.get("verified"):
                        repairs_by_claim[step.claim_id] = step.repair_data
                        if rep_rev not in repaired_texts:
                            repaired_texts.append(rep_rev)

        for c in claims:
            txt = c.normalized_text or c.original_text
            c_assmt = assmts.get(c.claim_id)
            if c.claim_id in repairs_by_claim:
                rep_rev = repairs_by_claim[c.claim_id].get("proposed_revision")
                sup_ids = set(repairs_by_claim[c.claim_id].get("supporting_evidence_ids", []))
                for p in evidence_store.get(c.claim_id, []):
                    if p.doc_title and (not sup_ids or p.evidence_id in sup_ids):
                        cited_sources.add(p.doc_title)
            elif c.state == ClaimState.VERIFIED:
                verified_texts.append(txt)
                provenance = c.evidence_provenance or {}
                supporting_spans = provenance.get("supporting_spans", {})
                supporting_ids = set(supporting_spans.keys()) if supporting_spans else None
                for p in evidence_store.get(c.claim_id, []):
                    if p.doc_title:
                        if supporting_ids is None or p.evidence_id in supporting_ids:
                            cited_sources.add(p.doc_title)
            elif c_assmt and c_assmt.label in (AssessmentLabel.CONTRADICTED, AssessmentLabel.CONFLICTED):
                contradicted_texts.append(txt)
                uncertainty_notes.append(f"Contradicted: '{txt}'")
            else:
                unresolved_texts.append(txt)
                uncertainty_notes.append(f"Unresolved: '{txt}'")

        import re
        clean_cited_sources = sorted([
            s for s in cited_sources
            if s and s not in ("Document Title", "Title") and not re.match(r'^Passage\s+\d+$', s, flags=re.IGNORECASE)
        ])
        sources_str = ", ".join(f"[{s}]" for s in clean_cited_sources) if clean_cited_sources else "[Evidence]"

        all_supported_statements = verified_texts + repaired_texts
        if all_supported_statements:
            final_answer = f"According to {sources_str}, " + " ".join(all_supported_statements)
            if contradicted_texts and not repaired_texts:
                final_answer += f" (Contradicted statements were excluded)."
            if unresolved_texts:
                final_answer += f" However, further details regarding {', '.join(unresolved_texts)} remain unverified."
        else:
            final_answer = "The available evidence does not contain sufficient verified information to establish an answer."
            if contradicted_texts:
                final_answer += " Earlier proposed statements were found to be contradicted."

        elapsed = time.monotonic() - start_time
        cost = VerificationCost(
            model_calls=self.default_cost.model_calls,
            prompt_tokens=self.default_cost.prompt_tokens,
            completion_tokens=self.default_cost.completion_tokens,
            total_tokens=self.default_cost.total_tokens,
            latency_seconds=elapsed,
            model_id="mock_synthesizer",
        )

        return SynthesisResult(
            question=question,
            initial_draft_answer=initial_draft_answer,
            final_answer=final_answer,
            verified_claims=verified_texts,
            unresolved_claims=unresolved_texts,
            contradicted_claims=contradicted_texts,
            cited_sources=clean_cited_sources,
            uncertainty_notes=uncertainty_notes,
            is_grounded=len(verified_texts) > 0 and len(clean_cited_sources) > 0,
            has_unresolved_limitations=len(unresolved_texts) > 0 or len(contradicted_texts) > 0 or len(repaired_texts) > 0,
            cost=cost,
            synthesis_method="mock_grounded",
        )
