#!/usr/bin/env python3
"""Stage D.5 — Warm-Start LinUCB Training on train[100:300].

Trains the ContextualMABPolicy (LinUCB) on 200 HotpotQA questions using:
  - Live Qdrant retrieval (collection: hotpotqa_bge-large-en-v1.5, port 6333)
  - Live llama.cpp assessment (port 8080, Llama-3.2-1B-Instruct-Q4_K_M.gguf)
  - ProxyVerificationRewardModel (zero gold annotations)

Saves checkpoint to experiments/runs/linucb_warmstart_train100_300.json.

Training boundary: train[100:300] (200 questions).
Evaluation boundary: train[800:1000] — NEVER accessed by this script.

Usage:
    python scripts/run_warmstart_training.py [--num-questions N] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure robust UTF-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.claims.models import (
    ClaimRecord,
    ClaimState,
    ClaimType,
    DecompositionStatus,
)
from src.claims.decomposer import RuleBasedDecomposer
from src.claims.filter import ContentBearingFilter
from src.claims.triage import ClaimTriageScorer
from src.controller.budget import BudgetTracker, PerClaimBudget, VerificationBudget
from src.controller.controller import VerificationController
from src.controller.interfaces import BaseRepairer, MockRepairer
from src.controller.mab_policy import ContextualMABPolicy, MABPolicyConfig
from src.controller.models import ControllerAction
from src.controller.reward import ProxyVerificationRewardModel
from src.controller.retriever import QdrantRetriever
from src.evidence.llama_assessor import LlamaAssessor
from src.evidence.models import EvidencePassage, VerificationCost

# Constants
DEFAULT_COLLECTION = "hotpotqa_bge-large-en-v1.5"
DEFAULT_LLAMA_URL = "http://127.0.0.1:8080/v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
TRAINING_SPLIT = "train[100:300]"
CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "runs"
CHECKPOINT_FILE = CHECKPOINT_DIR / "linucb_warmstart_train100_300.json"

# Budget configuration (per-question)
MAX_STEPS_PER_QUESTION = 5
MAX_MODEL_CALLS_PER_QUESTION = 8


def build_generation_prompt(question: str, passages: List[EvidencePassage]) -> str:
    """Build RAG generation prompt from question and retrieved passages."""
    context_parts = []
    for i, p in enumerate(passages):
        context_parts.append(f"[{i + 1}] {p.doc_title}: {p.source_text}")
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


def generate_draft_answer(
    question: str,
    passages: List[EvidencePassage],
    base_url: str = DEFAULT_LLAMA_URL,
) -> tuple:
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
        max_tokens=256,
    )
    answer = resp.choices[0].message.content or ""
    elapsed = time.monotonic() - start
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    cost = VerificationCost(
        model_calls=1,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_seconds=elapsed,
        model_id=model_name,
    )
    return answer, cost


def run_warmstart_training(
    num_questions: int = 200,
    seed: int = 42,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute warm-start training loop.

    Args:
        num_questions: Number of questions from train[100:300] to train on.
        seed: Random seed for reproducibility.
        resume_from: Optional path to existing checkpoint to resume training.
    """
    random.seed(seed)
    print(f"[D.5] Warm-Start LinUCB Training")
    print(f"    Split: {TRAINING_SPLIT}")
    print(f"    Seed: {seed}")
    print(f"    Questions: {num_questions}")
    print(f"    Budget: {MAX_STEPS_PER_QUESTION} steps, {MAX_MODEL_CALLS_PER_QUESTION} model calls per question")
    print()

    # Load dataset
    from datasets import load_dataset
    print(f"Loading HotpotQA split '{TRAINING_SPLIT}'...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=TRAINING_SPLIT)
    data = list(ds)[:num_questions]
    print(f"  Loaded {len(data)} questions.")

    # Initialize services
    print("Initializing retriever (Qdrant)...")
    retriever = QdrantRetriever(
        collection_name=DEFAULT_COLLECTION,
        model_name=DEFAULT_EMBEDDING_MODEL,
    )

    print("Initializing assessor (LlamaAssessor with context budgeting)...")
    assessor = LlamaAssessor(
        base_url=DEFAULT_LLAMA_URL,
        max_passages=4,
        max_passage_chars=900,
    )

    repairer: BaseRepairer = MockRepairer()
    content_filter = ContentBearingFilter()
    decomposer = RuleBasedDecomposer()
    triage_scorer = ClaimTriageScorer()

    # Initialize or resume LinUCB policy
    training_log: List[Dict[str, Any]] = []
    cumulative_reward = 0.0
    start_idx = 0

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        mab_policy = ContextualMABPolicy.load_checkpoint(resume_from, frozen=False)
        start_idx = mab_policy.checkpoint_metadata.get("questions_completed", 0)
        cumulative_reward = float(mab_policy.checkpoint_metadata.get("cumulative_reward", 0.0))
        print(f"  --> Resuming from question index {start_idx} (completed {start_idx}/{len(data)}), cumulative reward = {cumulative_reward:.2f}")

        # Load existing training log entries if present
        existing_log_file = CHECKPOINT_DIR / "linucb_warmstart_training_log.json"
        if existing_log_file.exists():
            try:
                with open(existing_log_file, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    training_log = old_data.get("per_question_log", [])[:start_idx]
                    print(f"  --> Loaded {len(training_log)} existing question log records.")
            except Exception as e:
                print(f"  --> Warning: could not load existing log: {e}")
    else:
        mab_policy = ContextualMABPolicy(
            config=MABPolicyConfig(
                alpha=0.5,
                frozen=False,
                random_seed=seed,
            ),
            reward_model=ProxyVerificationRewardModel(),
        )
    mab_policy.training_split = TRAINING_SPLIT
    mab_policy.random_seed = seed

    # Budget config
    global_budget = VerificationBudget(
        max_steps=MAX_STEPS_PER_QUESTION,
        max_model_calls=MAX_MODEL_CALLS_PER_QUESTION,
    )
    per_claim_budget = PerClaimBudget(
        max_retrievals=2,
        max_verifications=2,
        max_repairs=1,
    )

    # Training loop tracking
    total_steps_all = 0
    total_errors = 0
    start_wall = time.monotonic()

    for q_idx in range(start_idx, len(data)):
        q_item = data[q_idx]
        q_id = str(q_item.get("id", f"q_{q_idx}"))
        question = str(q_item.get("question", ""))
        q_start = time.monotonic()

        print(f"\n[{q_idx + 1}/{len(data)}] Q: {question[:80]}... (id={q_id})")

        try:
            # Phase 1: Initial retrieval
            dummy_claim = ClaimRecord(
                claim_id="init_query",
                source_sentence_id="s0",
                original_text=question,
                normalized_text=question,
                claim_type=ClaimType.FACTUAL_RELATION,
                decomposition_status=DecompositionStatus.ATOMIC,
            )
            initial_passages, init_ret_cost = retriever.retrieve(dummy_claim, top_k=5)

            # Phase 2: Draft answer generation
            draft_answer, init_gen_cost = generate_draft_answer(question, initial_passages)

            # Phase 3: Decompose into atomic claims
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

            # Phase 4: LinUCB verification (TRAINING MODE — parameters update)
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

            ctrl = VerificationController(
                retriever=retriever,
                assessor=assessor,
                repairer=repairer,
                policy=mab_policy,
                global_budget=global_budget,
                per_claim_budget=per_claim_budget,
            )
            trace = ctrl.run(
                claims=c_clones,
                initial_evidence=init_ev_store,
                run_id=f"warmstart_{q_id}",
            )

            # Compute question-level stats
            q_elapsed = time.monotonic() - q_start
            q_steps = trace.total_steps
            total_steps_all += q_steps

            q_errors = sum(1 for s in trace.steps if s.error is not None)
            total_errors += q_errors

            verified = sum(1 for c in c_clones if c.state == ClaimState.VERIFIED)
            unresolved = sum(1 for c in c_clones if c.state in (ClaimState.UNRESOLVED, ClaimState.PENDING))

            # Approximate reward from policy feedback updates
            q_reward = 0.0
            for step in trace.steps:
                if step.action != ControllerAction.STOP:
                    # Approximate per-step reward: check the step outcome
                    if step.error:
                        q_reward -= 0.8
                    elif step.assessment_label:
                        if step.assessment_label == "SUPPORTED" and (step.assessment_confidence or 0) >= 0.6:
                            q_reward += 0.8
                        elif step.assessment_label in ("CONTRADICTED", "CONFLICTED"):
                            q_reward += 0.6
                        elif step.assessment_label == "INSUFFICIENT":
                            q_reward -= 0.1
                    elif step.action == ControllerAction.RETRIEVE:
                        q_reward += 0.2  # Approximate novel retrieval

            cumulative_reward += q_reward

            # Track arm counts and theta norms
            arm_counts = {
                ControllerAction.RETRIEVE.value: sum(1 for s in trace.steps if s.action == ControllerAction.RETRIEVE),
                ControllerAction.VERIFY.value: sum(1 for s in trace.steps if s.action == ControllerAction.VERIFY),
                ControllerAction.REPAIR.value: sum(1 for s in trace.steps if s.action == ControllerAction.REPAIR),
            }
            theta_norms = {
                act.value: round(math.sqrt(sum(x * x for x in arm.get_theta())), 4)
                for act, arm in mab_policy.arms.items()
            }

            q_record = {
                "question_index": q_idx,
                "question_id": q_id,
                "question": question[:200],
                "draft_answer": draft_answer[:200],
                "num_claims": len(c_clones),
                "steps_taken": q_steps,
                "stopping_reason": trace.stopping_reason.value,
                "verified": verified,
                "unresolved": unresolved,
                "errors": q_errors,
                "arm_selections": arm_counts,
                "theta_norms": theta_norms,
                "approx_reward": round(q_reward, 4),
                "cumulative_reward": round(cumulative_reward, 4),
                "latency_seconds": round(q_elapsed, 2),
                "episodes_trained": mab_policy.episodes_trained,
                "total_feedback_updates": mab_policy.total_feedback_updates,
            }
            training_log.append(q_record)

            print(f"  Steps: {q_steps} | Claims: {len(c_clones)} (V:{verified}/U:{unresolved}) | "
                  f"Arms: R{arm_counts['RETRIEVE']}/V{arm_counts['VERIFY']}/Rep{arm_counts['REPAIR']} | "
                  f"Reward~{q_reward:+.2f} | Cum:{cumulative_reward:.2f} | "
                  f"Updates: {mab_policy.total_feedback_updates} | "
                  f"Time: {q_elapsed:.1f}s")

            # Intermediate checkpoint and log flush every 50 questions
            if (q_idx + 1) % 50 == 0:
                intermediate_ckpt = CHECKPOINT_DIR / f"linucb_warmstart_checkpoint_q{q_idx + 1}.json"
                mab_policy.save_checkpoint(
                    filepath=intermediate_ckpt,
                    training_split=TRAINING_SPLIT,
                    random_seed=seed,
                    metadata={
                        "stage": "D.5",
                        "description": f"Intermediate checkpoint at Q{q_idx + 1}",
                        "questions_completed": q_idx + 1,
                        "episodes_trained": mab_policy.episodes_trained,
                        "total_feedback_updates": mab_policy.total_feedback_updates,
                        "cumulative_reward": round(cumulative_reward, 4),
                    },
                )
                print(f"  --> Saved intermediate checkpoint: {intermediate_ckpt.name}")

                # Save intermediate training log
                log_file = CHECKPOINT_DIR / "linucb_warmstart_training_log.json"
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "schema_version": "1.0.0",
                        "training_split": TRAINING_SPLIT,
                        "random_seed": seed,
                        "num_questions": len(training_log),
                        "per_question_log": training_log,
                        "timestamp": datetime.now().isoformat(),
                    }, f, indent=2)

        except Exception as exc:
            q_elapsed = time.monotonic() - q_start
            total_errors += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            training_log.append({
                "question_index": q_idx,
                "question_id": q_id,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                "latency_seconds": round(q_elapsed, 2),
            })

    # Training summary
    total_elapsed = time.monotonic() - start_wall
    print(f"\n{'='*70}")
    print(f"Training Complete")
    print(f"  Questions processed: {len(training_log)}")
    print(f"  Total steps: {total_steps_all}")
    print(f"  Total errors: {total_errors}")
    print(f"  Episodes trained: {mab_policy.episodes_trained}")
    print(f"  Total feedback updates: {mab_policy.total_feedback_updates}")
    print(f"  Cumulative reward: {cumulative_reward:.4f}")
    print(f"  Wall time: {total_elapsed:.1f}s")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    mab_policy.save_checkpoint(
        filepath=CHECKPOINT_FILE,
        training_split=TRAINING_SPLIT,
        random_seed=seed,
        metadata={
            "stage": "D.5",
            "description": "Warm-start LinUCB trained on train[100:300]",
            "num_questions": len(training_log),
            "total_steps": total_steps_all,
            "total_errors": total_errors,
            "cumulative_reward": round(cumulative_reward, 4),
            "wall_time_seconds": round(total_elapsed, 1),
            "training_completed": datetime.now().isoformat(),
        },
    )
    print(f"\nCheckpoint saved: {CHECKPOINT_FILE}")

    # Save training log
    log_file = CHECKPOINT_DIR / "linucb_warmstart_training_log.json"
    log_data = {
        "schema_version": "1.0.0",
        "training_split": TRAINING_SPLIT,
        "random_seed": seed,
        "num_questions": len(training_log),
        "budget_per_question": {
            "max_steps": MAX_STEPS_PER_QUESTION,
            "max_model_calls": MAX_MODEL_CALLS_PER_QUESTION,
        },
        "summary": {
            "episodes_trained": mab_policy.episodes_trained,
            "total_feedback_updates": mab_policy.total_feedback_updates,
            "total_steps": total_steps_all,
            "total_errors": total_errors,
            "cumulative_reward": round(cumulative_reward, 4),
            "wall_time_seconds": round(total_elapsed, 1),
        },
        "arm_stats": {
            act.value: {
                "update_count": arm.update_count,
                "theta": arm.get_theta(),
            }
            for act, arm in mab_policy.arms.items()
        },
        "per_question_log": training_log,
        "timestamp": datetime.now().isoformat(),
    }
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    print(f"Training log saved: {log_file}")

    return log_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage D.5: Warm-Start LinUCB Training")
    parser.add_argument("--num-questions", type=int, default=200,
                        help="Number of questions from train[100:300] to train on (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing checkpoint to resume training from")
    args = parser.parse_args()

    run_warmstart_training(
        num_questions=args.num_questions,
        seed=args.seed,
        resume_from=args.resume,
    )
