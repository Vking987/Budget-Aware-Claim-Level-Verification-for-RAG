"""Isolated evaluation harness for controlled policy comparison.

Executes a fair, controlled comparison across four verification allocation paradigms:
1. Policy A: Vanilla RAG baseline (no post-hoc verification).
2. Policy B: Static FIFO verification allocation.
3. Policy C: Deterministic CTS-prioritized sequential controller.
4. Policy D: Contextual LinUCB Multi-Armed Bandit controller.

CRITICAL METHODOLOGICAL INVARIANTS:
1. All verification policies receive the EXACT SAME initial draft answer,
   extracted claims, initial evidence passages, and per-question resource budget.
2. Draft answers are generated ONCE per question to isolate allocation effects.
3. Evaluates on designated held-out partition without leaking test labels to policy state.
4. LinUCB online updates use strictly runtime observable ProxyVerificationRewardModel;
   gold answers and supporting titles are quarantined strictly for offline evaluation.
5. Consistent 3-tier cost accounting: initial_cost, verification_cost, total_cost.
6. Offline reward logging evaluates actual per-step claims, assessments, and newly retrieved passages.
7. Always outputs to a new, uniquely timestamped file. Never overwrites existing runs.
8. Supports an offline --dry-run mode using mock retriever and assessor backends.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.claims.filter import ContentBearingFilter
from src.claims.decomposer import RuleBasedDecomposer
from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus
from src.claims.triage import ClaimTriageScorer
from src.controller.budget import BudgetTracker
from src.controller.controller import VerificationController
from src.controller.interfaces import BaseRepairer, BaseRetriever, LLMRepairer, MockRepairer, MockRetriever
from src.controller.mab_policy import ContextualMABPolicy, MABPolicyConfig
from src.controller.models import (
    ControllerAction,
    ControllerTrace,
    PerClaimBudget,
    RepairResult,
    StoppingReason,
    VerificationBudget,
)
from src.controller.policy import (
    ControllerPolicyConfig,
    SequentialDecisionPolicy,
    StaticAllocationPolicy,
    StaticCTSRankedPolicy,
)
from src.controller.retriever import QdrantRetriever
from src.controller.reward import (
    CorrectnessObservability,
    GoldAnnotatedRewardModel,
    ProxyVerificationRewardModel,
    RewardOutcome,
)
from src.evaluation.metrics import (
    compute_generation_metrics,
    compute_retrieval_metrics,
)
from src.evidence.assessor import BaseAssessor, MockDeterministicAssessor
from src.evidence.llama_assessor import LlamaAssessor
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost


DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_COLLECTION_NAME = f"hotpotqa_{DEFAULT_MODEL_NAME.split('/')[-1]}"
DEFAULT_LLAMA_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_SPLIT = "train[800:820]"  # Strictly held out from initial pilot (train[:100])
OUTPUT_DIR = REPO_ROOT / "experiments" / "runs"


def build_generation_prompt(question: str, passages: List[EvidencePassage], top_k: int = 5) -> str:
    """Build standardized question-answering prompt identical to vanilla_rag.py."""
    context_parts = []
    for i, p in enumerate(passages[:top_k]):
        context_parts.append(f"[Document {i + 1}]\n{p.source_text}")
    context = "\n\n".join(context_parts)
    return (
        "You are a question-answering assistant.\n\n"
        "Answer the question using only the provided context.\n\n"
        "If the context does not contain enough information to answer\n"
        "the question, say that the information is insufficient.\n\n"
        "Do not invent facts.\n\n"
        f"Context:\n\n{context}\n\n"
        f"Question:\n\n{question}\n\n"
        "Answer:\n"
    )


def generate_draft_answer_mock(question: str, passages: List[EvidencePassage]) -> str:
    """Deterministic mock answer generator for offline verification and testing."""
    if not passages:
        return "The provided information is insufficient to answer the question."
    first_title = passages[0].doc_title
    return f"Based on {first_title}, Arthur's Magazine was established in 1844 and published in Philadelphia."


def generate_draft_answer_llm(
    question: str,
    passages: List[EvidencePassage],
    base_url: str = DEFAULT_LLAMA_BASE_URL,
    max_tokens: int = 256,
) -> Tuple[str, VerificationCost]:
    """Generate initial draft answer using local Llama endpoint."""
    from openai import OpenAI
    start = time.monotonic()
    client = OpenAI(base_url=base_url, api_key="local", timeout=30.0)
    prompt = build_generation_prompt(question, passages)
    models = client.models.list()
    model_name = models.data[0].id if models.data else "llama_local"

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    answer = resp.choices[0].message.content or ""
    elapsed = time.monotonic() - start
    prompt_tokens = getattr(getattr(resp, "usage", None), "prompt_tokens", len(prompt.split()))
    completion_tokens = getattr(getattr(resp, "usage", None), "completion_tokens", len(answer.split()))

    cost = VerificationCost(
        model_calls=1,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_seconds=elapsed,
        model_id=model_name,
    )
    return answer.strip(), cost


def run_controlled_comparison(
    questions_data: List[Dict[str, Any]],
    split_name: str,
    dry_run: bool = False,
    max_steps: int = 5,
    max_model_calls: int = 8,
    output_dir: Optional[Path] = None,
    freeze_policy: bool = False,
    load_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute controlled comparison across all policies on identical initial states.

    Args:
        questions_data: List of question dicts containing id, question, gold_titles, answer.
        split_name: Dataset split identifier.
        dry_run: If True, uses mock retriever and mock assessor offline.
        max_steps: Verification step limit per question.
        max_model_calls: Verification model call limit per question.
        output_dir: Output directory for JSON trace artifacts.
        freeze_policy: If True, freezes LinUCB parameters during evaluation (Phase 2).
        load_checkpoint: Optional path to pre-trained LinUCB checkpoint JSON file.

    Returns:
        Structured experiment dictionary containing records and aggregate statistics.
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup service backends
    if dry_run:
        retriever: BaseRetriever = MockRetriever()
        assessor: BaseAssessor = MockDeterministicAssessor()
    else:
        retriever = QdrantRetriever(
            collection_name=DEFAULT_COLLECTION_NAME,
            model_name=DEFAULT_MODEL_NAME,
        )
        assessor = LlamaAssessor(base_url=DEFAULT_LLAMA_BASE_URL)

    repairer: BaseRepairer = MockRepairer() if dry_run else LLMRepairer(base_url=DEFAULT_LLAMA_BASE_URL)
    content_filter = ContentBearingFilter()
    decomposer = RuleBasedDecomposer()
    triage_scorer = ClaimTriageScorer()

    # Pre-configure budgets
    global_budget = VerificationBudget(
        max_steps=max_steps,
        max_model_calls=max_model_calls,
    )
    per_claim_budget = PerClaimBudget(
        max_retrievals=2,
        max_verifications=2,
        max_repairs=1,
    )

    # Contextual MAB policy: Uses strictly runtime-observable ProxyVerificationRewardModel
    # Gold ground truth is quarantined and NEVER given to the online bandit policy (Fix F-01)
    if load_checkpoint:
        mab_policy = ContextualMABPolicy.load_checkpoint(
            load_checkpoint,
            frozen=freeze_policy,
        )
    else:
        mab_policy = ContextualMABPolicy(
            config=MABPolicyConfig(alpha=0.5, frozen=freeze_policy),
            reward_model=ProxyVerificationRewardModel(),
        )

    records: List[Dict[str, Any]] = []

    for q_idx, q_item in enumerate(questions_data):
        q_id = str(q_item.get("id", f"q_{q_idx}"))
        question = str(q_item.get("question", ""))
        gold_answer = str(q_item.get("answer", ""))
        raw_gold_titles = q_item.get("supporting_facts", {}).get("title", []) if isinstance(q_item.get("supporting_facts"), dict) else q_item.get("gold_titles", [])
        gold_titles = sorted(list(set(raw_gold_titles)))

        # Offline evaluation reward model: Strictly quarantined for post-hoc analysis (Fix F-01)
        gold_eval_model = GoldAnnotatedRewardModel(
            gold_supporting_titles=set(gold_titles),
            gold_answer=gold_answer,
        )

        # ----------------------------------------------------------------------
        # Phase 1: Shared Initial Retrieval & Generation (IDENTICAL ACROSS POLICIES)
        # ----------------------------------------------------------------------
        # Initial retrieval for draft answer
        dummy_claim = ClaimRecord(
            claim_id="init_query",
            source_sentence_id="s0",
            original_text=question,
            normalized_text=question,
            claim_type=ClaimType.FACTUAL_RELATION,
            decomposition_status=DecompositionStatus.ATOMIC,
        )
        initial_passages, init_ret_cost = retriever.retrieve(dummy_claim, top_k=5)

        # Initial draft answer generation
        if dry_run:
            draft_answer = generate_draft_answer_mock(question, initial_passages)
            init_gen_cost = VerificationCost(model_calls=0, latency_seconds=0.001)
        else:
            draft_answer, init_gen_cost = generate_draft_answer_llm(question, initial_passages)

        # Decompose draft answer into atomic claims using filter and decomposer
        sentence_records = content_filter.process(draft_answer)
        extracted_claims: List[ClaimRecord] = []
        for sent in sentence_records:
            if sent.is_content_bearing:
                extracted_claims.extend(decomposer.decompose(sent))

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
            c.triage_score = triage_scorer.score_claim(c, context_text=draft_answer)

        # Standardized Initial Cost Accounting (Fix F-02)
        init_cost_dict = {
            "model_calls": (init_ret_cost.model_calls or 0) + (init_gen_cost.model_calls or 0),
            "prompt_tokens": (init_ret_cost.prompt_tokens or 0) + (init_gen_cost.prompt_tokens or 0),
            "completion_tokens": (init_ret_cost.completion_tokens or 0) + (init_gen_cost.completion_tokens or 0),
            "total_tokens": (init_ret_cost.total_tokens or 0) + (init_gen_cost.total_tokens or 0),
            "latency_seconds": round((init_ret_cost.latency_seconds or 0.0) + (init_gen_cost.latency_seconds or 0.0), 4),
        }

        # ----------------------------------------------------------------------
        # Phase 2: Policy A — Vanilla RAG Baseline (No post-hoc verification)
        # ----------------------------------------------------------------------
        retrieved_titles = [p.doc_title for p in initial_passages]
        vanilla_gen_metrics = compute_generation_metrics(draft_answer, gold_answer)
        vanilla_ret_metrics = compute_retrieval_metrics(retrieved_titles, gold_titles)

        vanilla_record = {
            "policy_name": "VanillaRAG",
            "final_answer": draft_answer,
            "generation_metrics": vanilla_gen_metrics,
            "retrieval_metrics": vanilla_ret_metrics,
            "steps_taken": 0,
            "stopping_reason": "NO_VERIFICATION",
            "claims_summary": {
                "total": len(extracted_claims),
                "verified": 0,
                "unresolved": len(extracted_claims),
                "deferred": 0,
            },
            # Consistent 3-tier cost accounting (Fix F-02)
            "initial_cost": dict(init_cost_dict),
            "verification_cost": {
                "model_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_seconds": 0.0,
            },
            "total_cost": dict(init_cost_dict),
            "rewards": [],
        }

        # ----------------------------------------------------------------------
        # Helper: Execute a verification policy on identical initial claims
        # ----------------------------------------------------------------------
        def _execute_verification_policy(
            policy_obj: Any,
            p_name: str,
            custom_global_budget: Optional[VerificationBudget] = None,
            custom_per_claim_budget: Optional[PerClaimBudget] = None,
            is_reference: bool = False,
        ) -> Dict[str, Any]:
            # Deepcopy claims to maintain strict isolation
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
            init_ev_store = {c.claim_id: list(initial_passages) for c in c_clones}

            run_global_budget = custom_global_budget or global_budget
            run_per_claim_budget = custom_per_claim_budget or per_claim_budget

            ctrl = VerificationController(
                retriever=retriever,
                assessor=assessor,
                repairer=repairer,
                policy=policy_obj,
                global_budget=run_global_budget,
                per_claim_budget=run_per_claim_budget,
            )
            trace = ctrl.run(
                claims=c_clones,
                initial_evidence=init_ev_store,
                run_id=f"{p_name}_{q_id}",
            )

            # Reconstruct per-step offline gold rewards faithfully (Fix F-03)
            claim_by_id = {c.claim_id: c for c in c_clones}
            cumulative_passages_by_claim: Dict[str, List[EvidencePassage]] = {
                c.claim_id: list(initial_passages) for c in c_clones
            }
            reward_outcomes = []

            for step in trace.steps:
                if step.action == ControllerAction.STOP:
                    continue

                step_claim = claim_by_id.get(step.claim_id)
                step_action = step.action
                step_error = step.error
                action_failed = step_error is not None

                # Reconstruct marginal passages if RETRIEVE
                new_p: List[EvidencePassage] = []
                if step_action == ControllerAction.RETRIEVE and step.retrieved_passages:
                    new_p = [
                        EvidencePassage(
                            evidence_id=rp["evidence_id"],
                            doc_title=rp["doc_title"],
                            source_text=rp["source_text"],
                            retrieval_score=rp.get("retrieval_score"),
                            metadata=rp.get("metadata"),
                        )
                        for rp in step.retrieved_passages
                    ]
                    if step.claim_id in cumulative_passages_by_claim:
                        cumulative_passages_by_claim[step.claim_id].extend(new_p)

                cum_p = cumulative_passages_by_claim.get(step.claim_id, list(initial_passages))

                # Reconstruct assessment if VERIFY
                step_assessment: Optional[EvidenceAssessment] = None
                if step_action == ControllerAction.VERIFY and step.assessment_data:
                    ad = step.assessment_data
                    step_assessment = EvidenceAssessment(
                        claim_id=ad["claim_id"],
                        evidence_ids=ad.get("evidence_ids", []),
                        supporting_spans=ad.get("supporting_spans", []),
                        label=AssessmentLabel(ad["label"]),
                        confidence=ad.get("confidence", 0.5),
                        rationale=ad.get("rationale", ""),
                        limitations=ad.get("limitations", ""),
                        cost=VerificationCost(**ad["cost"]) if ad.get("cost") else None,
                        timestamp=ad.get("timestamp", ""),
                        assessor_id=ad.get("assessor_id", "assessor"),
                    )

                # Reconstruct repair if REPAIR
                step_repair: Optional[RepairResult] = None
                if step_action == ControllerAction.REPAIR and step.repair_data:
                    rd = step.repair_data
                    step_repair = RepairResult(
                        claim_id=rd["claim_id"],
                        original_text=rd["original_text"],
                        proposed_revision=rd["proposed_revision"],
                        repair_reason=rd["repair_reason"],
                        supporting_evidence_ids=rd.get("supporting_evidence_ids", []),
                        confidence=rd.get("confidence", 0.5),
                        cost=VerificationCost(**rd["cost"]) if rd.get("cost") else None,
                    )

                if step_claim is None and not action_failed:
                    reward_outcomes.append({
                        "step_number": step.step_number,
                        "action": step.action.value,
                        "reward_status": "UNAVAILABLE",
                        "rationale": "Step missing target claim; reward not computed.",
                    })
                    continue

                rew = gold_eval_model.compute_reward(
                    claim=step_claim,
                    action=step_action,
                    new_passages=new_p,
                    cumulative_passages=cum_p,
                    passages=new_p if step_action == ControllerAction.RETRIEVE else cum_p,
                    assessment=step_assessment,
                    repair=step_repair,
                    cost=VerificationCost(**step.action_cost) if step.action_cost else None,
                    action_failed=action_failed,
                    failure_reason=step_error,
                )
                rew_dict = rew.to_dict()
                rew_dict["step_number"] = step.step_number
                rew_dict["claim_id"] = step.claim_id
                rew_dict["action"] = step.action.value
                reward_outcomes.append(rew_dict)

            # Consistent 3-tier cost accounting (Fix F-02)
            verif_cost_dict = {
                "model_calls": trace.total_cost.get("model_calls", 0),
                "prompt_tokens": trace.total_cost.get("prompt_tokens", 0),
                "completion_tokens": trace.total_cost.get("completion_tokens", 0),
                "total_tokens": trace.total_cost.get("total_tokens", 0),
                "latency_seconds": round(trace.total_cost.get("latency_seconds", 0.0), 4),
            }
            total_end_to_end_cost = {
                "model_calls": init_cost_dict["model_calls"] + verif_cost_dict["model_calls"],
                "prompt_tokens": init_cost_dict["prompt_tokens"] + verif_cost_dict["prompt_tokens"],
                "completion_tokens": init_cost_dict["completion_tokens"] + verif_cost_dict["completion_tokens"],
                "total_tokens": init_cost_dict["total_tokens"] + verif_cost_dict["total_tokens"],
                "latency_seconds": round(init_cost_dict["latency_seconds"] + verif_cost_dict["latency_seconds"], 4),
            }

            verified_count = sum(1 for c in c_clones if c.state == ClaimState.VERIFIED)
            unresolved_count = sum(1 for c in c_clones if c.state in (ClaimState.UNRESOLVED, ClaimState.PENDING))
            deferred_count = sum(1 for c in c_clones if c.state == ClaimState.DEFERRED)

            exceeded_budget = False
            if is_reference:
                exceeded_budget = (
                    trace.total_steps > (global_budget.max_steps or 0)
                    or verif_cost_dict["model_calls"] > (global_budget.max_model_calls or 0)
                )

            return {
                "policy_name": p_name,
                "is_reference_policy": is_reference,
                "budget_constrained": not is_reference,
                "exceeded_experimental_budget": exceeded_budget,
                "reference_note": (
                    "Exhaustive verification quality ceiling: operates with unconstrained resource budget to observe asymptotic performance limit; not a budget-feasible competitor."
                    if is_reference else None
                ),
                "final_answer": draft_answer,  # Without Stage C repair/synthesis, answer text reflects draft
                "generation_metrics": vanilla_gen_metrics,
                "steps_taken": trace.total_steps,
                "stopping_reason": trace.stopping_reason.value,
                "claims_summary": {
                    "total": len(c_clones),
                    "verified": verified_count,
                    "unresolved": unresolved_count,
                    "deferred": deferred_count,
                },
                "initial_cost": dict(init_cost_dict),
                "verification_cost": verif_cost_dict,
                "total_cost": total_end_to_end_cost,
                "rewards": reward_outcomes,
                "trace_steps_count": len(trace.steps),
            }

        # ----------------------------------------------------------------------
        # Phase 3: Execute Verification Paradigms (B, C, D, E, F)
        # ----------------------------------------------------------------------
        # Policy B: Static FIFO Allocation
        static_fifo_policy = StaticAllocationPolicy(ControllerPolicyConfig())
        static_fifo_record = _execute_verification_policy(static_fifo_policy, "StaticFIFO")

        # Policy C: Static CTS-Ranked Allocation (AUD-04)
        static_cts_policy = StaticCTSRankedPolicy(ControllerPolicyConfig())
        static_cts_record = _execute_verification_policy(static_cts_policy, "StaticCTSRanked")

        # Policy D: Deterministic CTS Controller (Dynamic Reallocation)
        cts_policy = SequentialDecisionPolicy(ControllerPolicyConfig())
        cts_record = _execute_verification_policy(cts_policy, "DeterministicCTS")

        # Policy E: Contextual LinUCB MAB Controller
        # AUD-07 Fix: ctrl.run() already finalizes the episode via mark_episode_end(run_identifier).
        # Extraneous second call removed.
        mab_record = _execute_verification_policy(mab_policy, "ContextualMAB_LinUCB")

        # Policy F: Exhaustive Verification Reference Ceiling (AUD-04)
        exhaustive_budget = VerificationBudget(
            max_steps=50,
            max_model_calls=50,
        )
        exhaustive_per_claim = PerClaimBudget(
            max_retrievals=3,
            max_verifications=3,
            max_repairs=1,
        )
        exhaustive_policy = SequentialDecisionPolicy(ControllerPolicyConfig())
        exhaustive_record = _execute_verification_policy(
            exhaustive_policy,
            "ExhaustiveReference",
            custom_global_budget=exhaustive_budget,
            custom_per_claim_budget=exhaustive_per_claim,
            is_reference=True,
        )

        # Combine question record
        question_entry = {
            "index": q_idx,
            "id": q_id,
            "question": question,
            "gold_titles": gold_titles,
            "gold_answer": gold_answer,
            "initial_draft_answer": draft_answer,
            "extracted_claims": [
                {
                    "claim_id": c.claim_id,
                    "text": c.original_text,
                    "type": c.claim_type.value,
                    "triage_score": c.triage_score,
                }
                for c in extracted_claims
            ],
            "initial_retrieval": [
                {"title": p.doc_title, "score": p.retrieval_score}
                for p in initial_passages
            ],
            "policies": {
                "vanilla_rag": vanilla_record,
                "static_fifo": static_fifo_record,
                "static_cts_ranked": static_cts_record,
                "deterministic_cts": cts_record,
                "contextual_mab": mab_record,
                "exhaustive_reference": exhaustive_record,
            },
        }
        records.append(question_entry)

    # --------------------------------------------------------------------------
    # Aggregate Statistics (Fix F-04: Multi-Dimensional Separate Aggregation)
    # --------------------------------------------------------------------------
    def _aggregate_policy(p_key: str) -> Dict[str, Any]:
        p_records = [r["policies"][p_key] for r in records]
        n = len(p_records) or 1

        # 1. Answer Generation Metrics (Macro EM and Token F1)
        macro_em = sum(p["generation_metrics"]["exact_match"] for p in p_records) / n
        macro_f1 = sum(p["generation_metrics"]["f1"] for p in p_records) / n

        # 2. Resource Consumption Metrics (Separated into Initial, Verification, and Total)
        avg_verif_steps = sum(p["steps_taken"] for p in p_records) / n
        avg_verif_calls = sum(p["verification_cost"].get("model_calls", 0) for p in p_records) / n
        avg_verif_tokens = sum(p["verification_cost"].get("total_tokens", 0) for p in p_records) / n
        avg_verif_latency = sum(p["verification_cost"].get("latency_seconds", 0.0) for p in p_records) / n

        avg_total_calls = sum(p["total_cost"].get("model_calls", 0) for p in p_records) / n
        avg_total_tokens = sum(p["total_cost"].get("total_tokens", 0) for p in p_records) / n
        avg_total_latency = sum(p["total_cost"].get("latency_seconds", 0.0) for p in p_records) / n

        # 3. Claim Verification State Metrics
        tot_claims = sum(p["claims_summary"].get("total", 0) for p in p_records)
        tot_verified = sum(p["claims_summary"].get("verified", 0) for p in p_records)
        tot_unresolved = sum(p["claims_summary"].get("unresolved", 0) for p in p_records)
        tot_deferred = sum(p["claims_summary"].get("deferred", 0) for p in p_records)
        coverage = (tot_verified / tot_claims) if tot_claims > 0 else 0.0
        unsupported_rate = ((tot_unresolved + tot_deferred) / tot_claims) if tot_claims > 0 else 0.0
        unresolved_rate = (tot_unresolved / tot_claims) if tot_claims > 0 else 0.0
        deferred_rate = (tot_deferred / tot_claims) if tot_claims > 0 else 0.0

        # 4. Offline Diagnostic Quality Reward
        avg_offline_reward = 0.0
        q_rewards = [
            sum(rw.get("quality_reward", 0.0) for rw in p.get("rewards", []) if isinstance(rw.get("quality_reward"), (int, float)))
            for p in p_records
        ]
        if q_rewards:
            avg_offline_reward = sum(q_rewards) / n

        is_ref = (p_key == "exhaustive_reference")
        agg = {
            "num_questions": n,
            "is_reference_policy": is_ref,
            "budget_constrained": not is_ref,
            "answer_generation": {
                "macro_exact_match": round(macro_em, 4),
                "macro_token_f1": round(macro_f1, 4),
                "disclaimer": "Evaluates draft answer; answer repair/synthesis scheduled for Stage C.",
            },
            "claim_verification_state": {
                "total_claims": tot_claims,
                "total_verified": tot_verified,
                "total_unresolved": tot_unresolved,
                "total_deferred": tot_deferred,
                "verification_coverage": round(coverage, 4),
                "unsupported_claim_rate": round(unsupported_rate, 4),
                "unresolved_rate": round(unresolved_rate, 4),
                "deferred_rate": round(deferred_rate, 4),
                "disclaimer": (
                    "Operational system states: VERIFIED indicates local assessor SUPPORTED (conf>=0.60). "
                    "Unsupported claims (unresolved + deferred) reflect budget limits, NOT factual contradiction or falsity."
                ),
            },
            "ground_truth_alignment": {
                "status": "NOT_MEASURABLE_WITHOUT_HUMAN_ANNOTATION",
                "annotation_protocol": "docs/annotation_protocol.md",
                "taxonomy": ["GT-SUPPORTED", "GT-CONTRADICTED", "GT-INSUFFICIENT", "GT-UNVERIFIABLE", "GT-AMBIGUOUS"],
                "note": (
                    "Corpus-grounded assessor judgments do not prove open-world factual ground truth. "
                    "Full claim-level precision, recall, and FAR require human adjudication per docs/annotation_protocol.md "
                    "and are not automatically computable from HotpotQA question-level labels."
                ),
            },
            "verification_cost": {
                "avg_steps": round(avg_verif_steps, 2),
                "avg_model_calls": round(avg_verif_calls, 2),
                "avg_total_tokens": round(avg_verif_tokens, 2),
                "avg_latency_seconds": round(avg_verif_latency, 4),
            },
            "total_end_to_end_cost": {
                "avg_model_calls": round(avg_total_calls, 2),
                "avg_total_tokens": round(avg_total_tokens, 2),
                "avg_latency_seconds": round(avg_total_latency, 4),
            },
            "offline_reward_diagnostic": {
                "avg_cumulative_quality_reward": round(avg_offline_reward, 4),
                "disclaimer": "Offline evaluation signal strictly separated from answer quality and policy decisions.",
            },
        }

        if is_ref:
            agg["reference_disclaimer"] = (
                "NON-COMPETITIVE REFERENCE CEILING: Demonstrates asymptotic quality when budget constraints "
                "are relaxed. Cannot be compared directly with budget-constrained policies without accounting for the cost disparity."
            )

        # Include retrieval metrics for Vanilla RAG
        if p_key == "vanilla_rag" and p_records and p_records[0].get("retrieval_metrics"):
            ret_metrics_keys = ["hit_1", "hit_3", "hit_5", "hit_10", "recall_1", "recall_3", "recall_5", "recall_10", "both_1", "both_3", "both_5", "both_10"]
            agg["initial_retrieval_metrics"] = {
                k: round(sum(p["retrieval_metrics"].get(k, 0.0) for p in p_records) / n, 4)
                for k in ret_metrics_keys
                if k in p_records[0]["retrieval_metrics"]
            }

        return agg

    # Discover actual served LLM model ID if live
    actual_llm_model_id = None
    if not dry_run:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=DEFAULT_LLAMA_BASE_URL, api_key="local", timeout=5.0)
            models_list = client.models.list()
            if models_list.data:
                actual_llm_model_id = models_list.data[0].id
        except Exception:
            actual_llm_model_id = "unknown_live_llama"
    else:
        actual_llm_model_id = "mock_deterministic"

    backend_identity = {
        "is_live_execution": not dry_run,
        "retriever": {
            "implementation": retriever.__class__.__name__,
            "is_real": not dry_run,
            "status": "REAL_QDRANT" if not dry_run else "MOCK_RETRIEVER",
            "qdrant_collection": DEFAULT_COLLECTION_NAME if not dry_run else None,
            "embedding_model": DEFAULT_MODEL_NAME if not dry_run else None,
        },
        "draft_generator": {
            "implementation": "generate_draft_answer_llm" if not dry_run else "generate_draft_answer_mock",
            "is_real": not dry_run,
            "status": "REAL_LLAMA_CPP" if not dry_run else "MOCK_GENERATOR",
            "base_url": DEFAULT_LLAMA_BASE_URL if not dry_run else None,
            "model_identifier": actual_llm_model_id,
        },
        "assessor": {
            "implementation": assessor.__class__.__name__,
            "is_real": not dry_run,
            "status": "REAL_LLAMA_ASSESSOR" if not dry_run else "MOCK_ASSESSOR",
            "base_url": DEFAULT_LLAMA_BASE_URL if not dry_run else None,
            "model_identifier": actual_llm_model_id,
        },
        "repairer": {
            "implementation": repairer.__class__.__name__,
            "is_real": not dry_run,
            "status": "REAL_LLM_REPAIRER" if not dry_run else "MOCK_REPAIRER",
            "note": "Repairs are re-verified by the controller; final_answer in this script is still the unchanged draft.",
        },
    }

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_run_id = f"comparison_{timestamp_str}_{uuid.uuid4().hex[:8]}"

    final_output = {
        "metadata": {
            "run_id": unique_run_id,
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "split": split_name,
            "num_questions": len(questions_data),
            "max_steps": max_steps,
            "max_model_calls": max_model_calls,
            "mab_frozen": mab_policy.frozen,
            "mab_checkpoint": load_checkpoint,
            "backend_identity": backend_identity,
            "disclaimer": (
                "CONTROLLED VERIFICATION COMPARISON: All verification policies "
                "received identical initial draft answers, extracted claims, "
                "initial passages, and resource budgets."
            ),
        },
        "backend_identity": backend_identity,
        "aggregate_summary": {
            "vanilla_rag": _aggregate_policy("vanilla_rag"),
            "static_fifo": _aggregate_policy("static_fifo"),
            "static_cts_ranked": _aggregate_policy("static_cts_ranked"),
            "deterministic_cts": _aggregate_policy("deterministic_cts"),
            "contextual_mab": _aggregate_policy("contextual_mab"),
            "exhaustive_reference": _aggregate_policy("exhaustive_reference"),
        },
        "records": records,
    }

    output_file = out_dir / f"{unique_run_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n[Comparison Harness] Completed. Structured results written to: {output_file}")
    return final_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controlled Policy Comparison Evaluation Harness")
    parser.add_argument("--num-questions", type=int, default=20, help="Number of questions to evaluate")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="Dataset split identifier")
    parser.add_argument("--dry-run", action="store_true", help="Execute offline with mock backends")
    parser.add_argument("--max-steps", type=int, default=5, help="Max verification steps per question")
    parser.add_argument("--max-calls", type=int, default=8, help="Max model calls per question")
    parser.add_argument(
        "--freeze-policy",
        action="store_true",
        help="Freeze contextual bandit policy parameters during evaluation (prohibits online weight/feedback updates)",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to serialized LinUCB checkpoint JSON to load before evaluation",
    )
    args = parser.parse_args()

    # If live run requested, check dependencies
    if not args.dry_run:
        from datasets import load_dataset
        print(f"Loading HotpotQA split '{args.split}'...")
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
        data = list(ds)[:args.num_questions]
    else:
        # Create minimal synthetic question data for dry-run
        data = [
            {
                "id": f"synthetic_{i}",
                "question": f"Which magazine was published first: Arthur's Magazine or Periodical {i}?",
                "answer": "Arthur's Magazine",
                "gold_titles": ["Arthur's Magazine"],
            }
            for i in range(args.num_questions)
        ]

    run_controlled_comparison(
        questions_data=data,
        split_name=args.split,
        dry_run=args.dry_run,
        max_steps=args.max_steps,
        max_model_calls=args.max_calls,
        freeze_policy=args.freeze_policy,
        load_checkpoint=args.load_checkpoint,
    )
