"""End-to-End RAG Verification Pipeline.

Connects:
1. Qdrant passage retrieval
2. Initial draft generation
3. Claim extraction and triage scoring
4. Policy-guided sequential verification (LinUCB, CTS, FIFO, Exhaustive)
5. Evidence-grounded final answer synthesis
6. Full 3-tier cost and citation accounting
"""

from __future__ import annotations

import os

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.claims.decomposer import RuleBasedDecomposer
from src.claims.filter import ContentBearingFilter
from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus
from src.claims.triage import ClaimTriageScorer
from src.controller.budget import BudgetTracker, PerClaimBudget, VerificationBudget
from src.controller.controller import VerificationController
from src.controller.interfaces import BaseRepairer, BaseRetriever, LLMRepairer, MockRepairer, MockRetriever
from src.controller.mab_policy import ContextualMABPolicy, MABPolicyConfig
from src.controller.models import ControllerAction, ControllerTrace
from src.controller.policy import (
    ControllerPolicyConfig,
    SequentialDecisionPolicy,
    StaticAllocationPolicy,
    StaticCTSRankedPolicy,
)
from src.controller.retriever import QdrantRetriever
from src.controller.reward import ProxyVerificationRewardModel
from src.evidence.assessor import BaseAssessor, MockDeterministicAssessor
from src.evidence.llama_assessor import LlamaAssessor
from src.evidence.models import EvidencePassage, VerificationCost
from src.synthesis.citations import audit_answer_sentences
from src.synthesis.models import SynthesisResult
from src.synthesis.synthesizer import BaseSynthesizer, LLMAnswerSynthesizer, MockAnswerSynthesizer


DEFAULT_WARMSTART_CHECKPOINT = Path("experiments/runs/linucb_warmstart_train100_300.json")


@dataclass
class PipelineResult:
    """Complete structured output from an end-to-end question verification run."""

    question: str
    initial_draft: str
    final_answer: str
    policy_name: str
    claims: List[Dict[str, Any]] = field(default_factory=list)
    evidence_passages: List[Dict[str, Any]] = field(default_factory=list)
    synthesis: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    initial_cost: Dict[str, Any] = field(default_factory=dict)
    verification_cost: Dict[str, Any] = field(default_factory=dict)
    synthesis_cost: Dict[str, Any] = field(default_factory=dict)
    total_cost: Dict[str, Any] = field(default_factory=dict)
    steps_taken: int = 0
    stopping_reason: str = ""
    answer_changed: bool = False
    is_grounded: bool = True
    has_unresolved_limitations: bool = False
    repairer_class: str = ""
    trace: Optional[ControllerTrace] = None
    draft_source: str = "generated"  # "generated" (LLM) or "user_supplied" (generation skipped)
    budget: Dict[str, Any] = field(default_factory=dict)
    answer_audit: List[Dict[str, Any]] = field(default_factory=list)  # per-sentence backing by verified claims (report only)
    answer_validation: Dict[str, Any] = field(default_factory=dict)  # enforced grounding validation outcome (LLM synthesizer)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "initial_draft": self.initial_draft,
            "final_answer": self.final_answer,
            "policy_name": self.policy_name,
            "answer_changed": self.answer_changed,
            "is_grounded": self.is_grounded,
            "has_unresolved_limitations": self.has_unresolved_limitations,
            "repairer_class": self.repairer_class,
            "draft_source": self.draft_source,
            "budget": self.budget,
            "answer_audit": self.answer_audit,
            "answer_validation": self.answer_validation,
            "steps_taken": self.steps_taken,
            "stopping_reason": self.stopping_reason,
            "claims": self.claims,
            "evidence_passages": self.evidence_passages,
            "synthesis": self.synthesis,
            "warnings": self.warnings,
            "trace": self.trace.to_dict() if self.trace else None,
            "cost": {
                "initial": self.initial_cost,
                "verification": self.verification_cost,
                "synthesis": self.synthesis_cost,
                "total": self.total_cost,
            },
        }


class RAGVerificationPipeline:
    """Coordinates end-to-end evidence retrieval, verification, and grounded answer synthesis."""

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        assessor: Optional[BaseAssessor] = None,
        repairer: Optional[BaseRepairer] = None,
        synthesizer: Optional[BaseSynthesizer] = None,
        content_filter: Optional[ContentBearingFilter] = None,
        decomposer: Optional[RuleBasedDecomposer] = None,
        triage_scorer: Optional[ClaimTriageScorer] = None,
        llm_client: Optional[Any] = None,
        dry_run: bool = False,
        collection_name: Optional[str] = None,
        long_answer: bool = False,
    ) -> None:
        self.dry_run = dry_run
        self.long_answer = long_answer  # multi-sentence draft (e.g. ALCE); default keeps the concise HotpotQA-style draft
        self.collection_name = collection_name
        self.retriever = retriever or (
            MockRetriever() if dry_run
            else QdrantRetriever(**({"collection_name": collection_name} if collection_name else {}))
        )
        self.assessor = assessor or (MockDeterministicAssessor() if dry_run else LlamaAssessor())
        self.repairer = repairer or (MockRepairer() if dry_run else LLMRepairer(client=llm_client))
        self.synthesizer = synthesizer or (MockAnswerSynthesizer() if dry_run else LLMAnswerSynthesizer(client=llm_client))
        self.content_filter = content_filter or ContentBearingFilter()
        self.decomposer = decomposer or RuleBasedDecomposer()
        self.triage_scorer = triage_scorer or ClaimTriageScorer()
        self._llm_client = llm_client

    def _generate_draft(self, question: str, passages: List[EvidencePassage]) -> Tuple[str, VerificationCost]:
        """Generate initial draft answer using retrieved passages."""
        if self.dry_run or isinstance(self.synthesizer, MockAnswerSynthesizer):
            if passages:
                draft = f"Based on retrieved documents, {question.replace('?', '').strip()} involves {passages[0].doc_title}."
            else:
                draft = "The available context does not provide sufficient information to answer the question."
            return draft, VerificationCost(model_calls=0, latency_seconds=0.001, model_id="mock_draft")

        # Real LLM call
        from openai import OpenAI
        start = time.monotonic()
        client = self._llm_client or OpenAI(base_url=os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1"), api_key="local", timeout=30.0)
        try:
            models = client.models.list()
            model_name = models.data[0].id if models.data else "local_model"
        except Exception:
            # llama-server's /v1/models shape differs from OpenAI's; it ignores the name anyway.
            model_name = "local_model"

        context_parts = [f"[{i+1}] {p.doc_title}: {p.source_text[:500]}" for i, p in enumerate(passages)]
        context_str = "\n\n".join(context_parts)
        if self.long_answer:
            instruction = (
                "Write a detailed answer of 4 to 6 sentences using only the context. Each sentence should state a "
                "distinct fact that answers part of the question. If the context is insufficient, say so."
            )
            max_out = 360
        else:
            instruction = "Answer the question concisely using the context. If uncertain, state the limitation."
            max_out = 256
        prompt = (
            f"You are a helpful question-answering assistant.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"{instruction}\nAnswer:"
        )

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_out,
            )
            ans = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            pt = getattr(usage, "prompt_tokens", 0) if usage else 0
            ct = getattr(usage, "completion_tokens", 0) if usage else 0
        except Exception as exc:
            raise RuntimeError(f"Draft generation failed against the LLM server: {exc}") from exc

        elapsed = time.monotonic() - start
        cost = VerificationCost(
            model_calls=1,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            latency_seconds=elapsed,
            model_id=model_name,
        )
        return ans.strip(), cost

    def _resolve_policy(
        self,
        policy_name: str,
        checkpoint_path: Optional[str | Path] = None,
        random_seed: int = 42,
    ) -> Tuple[Any, VerificationBudget, PerClaimBudget]:
        """Instantiate selected verification policy and corresponding budgets."""
        global_budget = VerificationBudget(max_steps=5, max_model_calls=8)
        per_claim_budget = PerClaimBudget(max_retrievals=2, max_verifications=2, max_repairs=1)

        p_norm = policy_name.lower().replace("_", "").replace("-", "").replace(" ", "")

        if "mab" in p_norm or "linucb" in p_norm:
            ckpt = Path(checkpoint_path) if checkpoint_path else DEFAULT_WARMSTART_CHECKPOINT
            if ckpt.exists():
                policy = ContextualMABPolicy.load_checkpoint(ckpt, frozen=True)
            else:
                policy = ContextualMABPolicy(
                    config=MABPolicyConfig(alpha=0.5, frozen=True, random_seed=random_seed),
                    reward_model=ProxyVerificationRewardModel(),
                )
            return policy, global_budget, per_claim_budget

        elif "staticcts" in p_norm or "ctsranked" in p_norm:
            return StaticCTSRankedPolicy(ControllerPolicyConfig()), global_budget, per_claim_budget

        elif "fifo" in p_norm:
            return StaticAllocationPolicy(ControllerPolicyConfig()), global_budget, per_claim_budget

        elif "deterministic" in p_norm or p_norm == "cts":
            return SequentialDecisionPolicy(ControllerPolicyConfig()), global_budget, per_claim_budget

        elif "exhaustive" in p_norm:
            ex_global = VerificationBudget(max_steps=50, max_model_calls=50)
            ex_per_claim = PerClaimBudget(max_retrievals=3, max_verifications=3, max_repairs=2)
            return SequentialDecisionPolicy(ControllerPolicyConfig()), ex_global, ex_per_claim

        elif "vanilla" in p_norm:
            # Zero-step budget
            zero_global = VerificationBudget(max_steps=0, max_model_calls=0)
            return SequentialDecisionPolicy(ControllerPolicyConfig()), zero_global, per_claim_budget

        raise ValueError(
            f"Unknown policy '{policy_name}'. Valid: ContextualMAB_LinUCB, DeterministicCTS, "
            "StaticCTSRanked, StaticFIFO, ExhaustiveReference, VanillaRAG."
        )

    def run(
        self,
        question: str,
        policy_name: str = "ContextualMAB_LinUCB",
        checkpoint_path: Optional[str | Path] = None,
        top_k: int = 5,
        draft_answer: Optional[str] = None,
        max_steps: Optional[int] = None,
        max_model_calls: Optional[int] = None,
    ) -> PipelineResult:
        """Execute end-to-end question answering and evidence-grounded verification.

        draft_answer: user-supplied draft. When given, initial LLM draft generation is SKIPPED (recorded as a
            zero-cost entry and a warning) and the supplied text goes through the unchanged claim extraction,
            triage, controller, repair/re-verification and synthesis stages. Initial retrieval still runs.
        max_steps / max_model_calls: optional overrides of the default global budget (5 steps / 8 calls).
        """
        warnings: List[str] = []
        if draft_answer is not None and not draft_answer.strip():
            raise ValueError("draft_answer was provided but is empty")

        # -------------------------------------------------------------
        # Step 1: Initial Retrieval
        # -------------------------------------------------------------
        dummy_claim = ClaimRecord(
            claim_id="init_query",
            source_sentence_id="s0",
            original_text=question,
            normalized_text=question,
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
        )
        try:
            initial_passages, init_ret_cost = self.retriever.retrieve(dummy_claim, top_k=top_k)
        except Exception as _retrieval_err:
            # Surface service-unavailable errors with a clear diagnostic instead of a raw traceback
            err_type = type(_retrieval_err).__name__
            raise RuntimeError(
                f"Initial retrieval failed ({err_type}: {_retrieval_err}). "
                "Ensure Qdrant is running on port 6333 and the collection is ingested. "
                "Run with --dry-run for offline mode."
            ) from _retrieval_err

        # -------------------------------------------------------------
        # Step 2: Initial Draft Answer Generation
        # -------------------------------------------------------------
        draft_source = "generated"
        if draft_answer is not None:
            draft_source = "user_supplied"
            draft_answer = draft_answer.strip()
            init_gen_cost = VerificationCost(model_calls=0, latency_seconds=0.0, model_id="user_supplied_draft")
            warnings.append("NOTE: Initial draft generation was SKIPPED; the draft was supplied by the user (--draft). "
                            "All later stages ran on the supplied text.")
        else:
            draft_answer, init_gen_cost = self._generate_draft(question, initial_passages)

        init_cost_dict = {
            "model_calls": (init_ret_cost.model_calls or 0) + (init_gen_cost.model_calls or 0),
            "prompt_tokens": (init_ret_cost.prompt_tokens or 0) + (init_gen_cost.prompt_tokens or 0),
            "completion_tokens": (init_ret_cost.completion_tokens or 0) + (init_gen_cost.completion_tokens or 0),
            "total_tokens": (init_ret_cost.total_tokens or 0) + (init_gen_cost.total_tokens or 0),
            "latency_seconds": round((init_ret_cost.latency_seconds or 0.0) + (init_gen_cost.latency_seconds or 0.0), 4),
        }

        # -------------------------------------------------------------
        # Step 3: Claim Extraction & Triage Scoring
        # -------------------------------------------------------------
        sentence_records = self.content_filter.process(draft_answer)
        extracted_claims: List[ClaimRecord] = []
        for sent in sentence_records:
            if sent.is_content_bearing:
                extracted_claims.extend(self.decomposer.decompose(sent))

        if not extracted_claims and draft_answer.strip():
            extracted_claims = [
                ClaimRecord(
                    claim_id="c_0",
                    source_sentence_id="s_0",
                    original_text=draft_answer.strip(),
                    normalized_text=draft_answer.strip(),
                    claim_type=ClaimType.FACTUAL_RELATION,
                    decomposition_status=DecompositionStatus.UNSPLIT_PRESERVED,
                    state=ClaimState.PENDING,
                )
            ]

        for c in extracted_claims:
            c.triage_score = self.triage_scorer.score_claim(c, context_text=draft_answer)

        # -------------------------------------------------------------
        # Step 4: Verification Controller Execution
        # -------------------------------------------------------------
        policy_obj, g_budget, pc_budget = self._resolve_policy(policy_name, checkpoint_path=checkpoint_path)
        if max_steps is not None:
            g_budget.max_steps = max_steps
        if max_model_calls is not None:
            g_budget.max_model_calls = max_model_calls

        c_clones = [
            ClaimRecord(
                claim_id=c.claim_id,
                source_sentence_id=c.source_sentence_id,
                original_text=c.original_text,
                normalized_text=c.normalized_text,
                claim_type=c.claim_type,
                decomposition_status=c.decomposition_status,
                state=ClaimState.PENDING,
                triage_score=c.triage_score,
            )
            for c in extracted_claims
        ]
        evidence_store = {c.claim_id: list(initial_passages) for c in c_clones}

        ctrl = VerificationController(
            retriever=self.retriever,
            assessor=self.assessor,
            repairer=self.repairer,
            policy=policy_obj,
            global_budget=g_budget,
            per_claim_budget=pc_budget,
        )

        trace = ctrl.run(
            claims=c_clones,
            initial_evidence=evidence_store,
            run_id=f"demo_{policy_name}",
        )

        verif_cost_dict = {
            "model_calls": trace.total_cost.get("model_calls", 0),
            "prompt_tokens": trace.total_cost.get("prompt_tokens", 0),
            "completion_tokens": trace.total_cost.get("completion_tokens", 0),
            "total_tokens": trace.total_cost.get("total_tokens", 0),
            "latency_seconds": round(trace.total_cost.get("latency_seconds", 0.0), 4),
        }

        # -------------------------------------------------------------
        # Step 5: Final Grounded Answer Synthesis
        # -------------------------------------------------------------
        synthesis_res = self.synthesizer.synthesize(
            question=question,
            initial_draft_answer=draft_answer,
            claims=c_clones,
            evidence_store=evidence_store,
            trace=trace,
        )

        synth_cost_dict = {
            "model_calls": synthesis_res.cost.model_calls or 0,
            "prompt_tokens": synthesis_res.cost.prompt_tokens or 0,
            "completion_tokens": synthesis_res.cost.completion_tokens or 0,
            "total_tokens": synthesis_res.cost.total_tokens or 0,
            "latency_seconds": round(synthesis_res.cost.latency_seconds or 0.0, 4),
        }

        total_cost_dict = {
            "model_calls": init_cost_dict["model_calls"] + verif_cost_dict["model_calls"] + synth_cost_dict["model_calls"],
            "prompt_tokens": init_cost_dict["prompt_tokens"] + verif_cost_dict["prompt_tokens"] + synth_cost_dict["prompt_tokens"],
            "completion_tokens": init_cost_dict["completion_tokens"] + verif_cost_dict["completion_tokens"] + synth_cost_dict["completion_tokens"],
            "total_tokens": init_cost_dict["total_tokens"] + verif_cost_dict["total_tokens"] + synth_cost_dict["total_tokens"],
            "latency_seconds": round(init_cost_dict["latency_seconds"] + verif_cost_dict["latency_seconds"] + synth_cost_dict["latency_seconds"], 4),
        }

        # Check for limitations and warnings
        verified_count = sum(1 for c in c_clones if c.state == ClaimState.VERIFIED)
        if verified_count == 0:
            warnings.append("WARNING: No claims could be verified against available evidence.")
        if synthesis_res.contradicted_claims:
            warnings.append(f"WARNING: {len(synthesis_res.contradicted_claims)} claim(s) were contradicted and excluded from the final answer.")
        if synthesis_res.has_unresolved_limitations:
            warnings.append("NOTE: Final answer contains unverified aspects due to budget or evidence limits.")

        # Enforced grounding (LLM synthesizer): the synthesizer already validated the answer it returned.
        answer_validation = {"enforced": bool(getattr(synthesis_res, "grounding_enforced", False)),
                             "violations": list(getattr(synthesis_res, "answer_violations", []) or []),
                             "polish_status": getattr(synthesis_res, "polish_status", "")}
        if answer_validation["violations"]:
            warnings.append(f"WARNING: final answer failed grounding validation: {answer_validation['violations'][:2]}")
        if answer_validation["polish_status"] == "llm_rejected_stitched":
            warnings.append("NOTE: the LLM rewrite of the answer failed grounding validation and was replaced by the "
                            "deterministic stitched answer built from the verified claims.")

        # Report-only lexical audit: which final-answer sentences are not backed by any verified claim?
        answer_audit = audit_answer_sentences(synthesis_res.final_answer, synthesis_res.verified_claims)
        _unbacked = [r for r in answer_audit if not r["backed"]]
        if _unbacked:
            warnings.append(
                f"WARNING: {len(_unbacked)} of {len(answer_audit)} sentence(s) in the final answer are not backed by any "
                "verified claim (content may come from raw passages, the draft, or an unverified repair); see answer_audit."
            )

        # Structure claim summaries for display
        claims_display = []
        for c in c_clones:
            assessment_label = None
            conf = None
            for step in trace.steps:
                if step.claim_id == c.claim_id and step.assessment_label:
                    assessment_label = step.assessment_label
                    conf = step.assessment_confidence

            claims_display.append({
                "claim_id": c.claim_id,
                "text": c.original_text,
                "state": c.state.value,
                "triage_score": round(c.triage_score, 4) if c.triage_score is not None else None,
                "assessment_label": assessment_label,
                "confidence": round(conf, 4) if conf is not None else None,
            })

        passages_display = [
            {
                "evidence_id": p.evidence_id,
                "doc_title": p.doc_title,
                "retrieval_score": round(p.retrieval_score, 4) if p.retrieval_score is not None else None,
                "snippet": p.source_text[:200] + ("..." if len(p.source_text) > 200 else ""),
                "text": p.source_text,
            }
            for p in initial_passages
        ]

        answer_changed = (draft_answer.strip() != synthesis_res.final_answer.strip())

        return PipelineResult(
            question=question,
            initial_draft=draft_answer,
            final_answer=synthesis_res.final_answer,
            policy_name=policy_name,
            claims=claims_display,
            evidence_passages=passages_display,
            synthesis=synthesis_res.to_dict(),
            warnings=warnings,
            initial_cost=init_cost_dict,
            verification_cost=verif_cost_dict,
            synthesis_cost=synth_cost_dict,
            total_cost=total_cost_dict,
            steps_taken=trace.total_steps,
            stopping_reason=trace.stopping_reason.value,
            answer_changed=answer_changed,
            is_grounded=synthesis_res.is_grounded,
            has_unresolved_limitations=synthesis_res.has_unresolved_limitations,
            repairer_class=type(self.repairer).__name__,
            trace=trace,
            draft_source=draft_source,
            answer_audit=answer_audit,
            answer_validation=answer_validation,
            budget={"max_steps": g_budget.max_steps, "max_model_calls": g_budget.max_model_calls,
                    "max_retrievals_per_claim": pc_budget.max_retrievals,
                    "max_verifications_per_claim": pc_budget.max_verifications,
                    "max_repairs_per_claim": pc_budget.max_repairs},
        )
