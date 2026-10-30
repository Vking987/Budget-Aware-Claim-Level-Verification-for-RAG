"""Isolated evaluation harness for reproducing and benchmarking the Vanilla RAG baseline.

This script executes the baseline retrieval and generation pipeline on a pilot of 100 questions
from the legacy reproduction split (train[:100]) without modifying existing baseline files
(src/vanilla_rag.py, src/retrieve.py, src/ingest.py).

Requirements satisfied:
- Verifies Qdrant and Llama services prior to execution.
- Computes Hit@K, true multi-hop Recall@K, Both@K, Exact Match (EM), and Token F1.
- Records per-question timings, context, predictions, errors, and aggregate statistics.
- Saves structured results to experiments/runs/baseline_pilot_100.json without overwriting existing files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.evaluation.metrics import (
    compute_generation_metrics,
    compute_retrieval_metrics,
)


# ==============================================================================
# Configuration Defaults (Exactly Matching Baseline)
# ==============================================================================

MODEL_NAME = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = f"hotpotqa_{MODEL_NAME.split('/')[-1]}"  # hotpotqa_bge-large-en-v1.5
TOP_K_RETRIEVAL = 10
TOP_K_GENERATION = 5
LLAMA_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_NUM_QUESTIONS = 100
DEFAULT_SPLIT = f"train[:{DEFAULT_NUM_QUESTIONS}]"
OUTPUT_DIR = REPO_ROOT / "experiments" / "runs"


# ==============================================================================
# Service Pre-flight Verification
# ==============================================================================

def check_qdrant_service(host: str = "localhost", port: int = 6333, collection: str = COLLECTION_NAME) -> bool:
    """Verify that Qdrant is running and the target collection exists."""
    print(f"[Pre-flight] Checking Qdrant connection at {host}:{port}...")
    try:
        client = QdrantClient(host=host, port=port, timeout=5)
        if not client.collection_exists(collection):
            print(f"[Error] Qdrant is running, but collection '{collection}' does not exist!")
            return False
        info = client.get_collection(collection)
        print(f"[Pre-flight] Qdrant OK. Collection '{collection}' found with {info.points_count} points.")
        return True
    except Exception as exc:
        print(f"[Error] Failed to connect to Qdrant at {host}:{port}: {exc}")
        return False


def check_llama_service(base_url: str = LLAMA_BASE_URL) -> str | None:
    """Verify that local Llama server is reachable and return the served model name."""
    print(f"[Pre-flight] Checking local Llama endpoint at {base_url}...")
    try:
        client = OpenAI(base_url=base_url, api_key="local", timeout=5.0)
        models = client.models.list()
        if not models.data:
            print("[Error] Llama server responded, but no models are available.")
            return None
        model_name = models.data[0].id
        print(f"[Pre-flight] Llama server OK. Serving model: '{model_name}'.")
        return model_name
    except Exception as exc:
        print(f"[Error] Failed to connect to Llama server at {base_url}: {exc}")
        return None


# ==============================================================================
# Prompt Construction (Identical to src/vanilla_rag.py)
# ==============================================================================

def build_prompt(question: str, retrieved_docs: List[Dict[str, Any]], top_k: int = TOP_K_GENERATION) -> str:
    """Construct prompt context and instructions exactly matching src/vanilla_rag.py."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:top_k]):
        text = doc["text"]
        context_parts.append(f"[Document {i + 1}]\n{text}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a question-answering assistant.

Answer the question using only the provided context.

If the context does not contain enough information to answer
the question, say that the information is insufficient.

Do not invent facts.

Context:

{context}

Question:

{question}

Answer:
"""
    return prompt


# ==============================================================================
# Main Runner
# ==============================================================================

def run_baseline_evaluation(
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    split_str: str = DEFAULT_SPLIT,
    output_filename: str = "baseline_pilot_100.json",
    dry_run: bool = False
) -> Dict[str, Any]:
    """Execute evaluation on the pilot split and write structured results."""
    # 1. Pre-flight checks
    if not check_qdrant_service():
        sys.exit(1)

    llm_model_name = None
    if not dry_run:
        llm_model_name = check_llama_service()
        if not llm_model_name:
            sys.exit(1)

    # 2. Load dataset
    print(f"\nLoading dataset split: hotpotqa/hotpot_qa (distractor, {split_str})...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split_str)
    print(f"Loaded {len(dataset)} questions for evaluation.")

    # 3. Load embedding model and clients
    print(f"Loading embedding model '{MODEL_NAME}'...")
    embedder = SentenceTransformer(MODEL_NAME)
    qdrant = QdrantClient(host="localhost", port=6333)

    llm_client = None
    if not dry_run:
        llm_client = OpenAI(base_url=LLAMA_BASE_URL, api_key="local")

    records: List[Dict[str, Any]] = []
    total_retrieval_time = 0.0
    total_generation_time = 0.0

    print("\nStarting evaluation loop...")
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating questions")):
        qid = example["id"]
        question = example["question"]
        gold_answer = example.get("answer", "")
        gold_titles = list(example["supporting_facts"]["title"])

        record: Dict[str, Any] = {
            "index": idx,
            "id": qid,
            "question": question,
            "gold_titles": gold_titles,
            "gold_answer": gold_answer,
            "retrieved_documents": [],
            "retrieval_metrics": {},
            "generation_metrics": {},
            "prediction": None,
            "prompt": None,
            "retrieval_time_sec": 0.0,
            "generation_time_sec": 0.0,
            "error": None,
        }

        # Retrieval phase
        try:
            t0 = time.perf_counter()
            query_vector = embedder.encode(question, normalize_embeddings=True).tolist()
            qdrant_points = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=TOP_K_RETRIEVAL
            ).points
            t_retrieval = time.perf_counter() - t0
            record["retrieval_time_sec"] = round(t_retrieval, 4)
            total_retrieval_time += t_retrieval

            retrieved_docs = []
            for rank, pt in enumerate(qdrant_points):
                retrieved_docs.append({
                    "rank": rank + 1,
                    "title": pt.payload.get("title", ""),
                    "score": round(pt.score, 5) if pt.score is not None else None,
                    "point_id": pt.id,
                    "text": pt.payload.get("text", "")
                })
            record["retrieved_documents"] = retrieved_docs

            retrieved_titles = [d["title"] for d in retrieved_docs]
            ret_metrics = compute_retrieval_metrics(retrieved_titles, gold_titles, cutoffs=(1, 3, 5, 10))
            record["retrieval_metrics"] = ret_metrics

        except Exception as exc:
            record["error"] = f"RetrievalError: {exc}"
            records.append(record)
            continue

        # Generation phase
        if not dry_run and llm_client is not None and llm_model_name is not None:
            prompt = build_prompt(question, retrieved_docs, top_k=TOP_K_GENERATION)
            record["prompt"] = prompt

            try:
                t0 = time.perf_counter()
                response = llm_client.chat.completions.create(
                    model=llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256
                )
                t_generation = time.perf_counter() - t0
                record["generation_time_sec"] = round(t_generation, 4)
                total_generation_time += t_generation

                prediction = response.choices[0].message.content
                record["prediction"] = prediction

                gen_metrics = compute_generation_metrics(prediction, gold_answer)
                record["generation_metrics"] = gen_metrics

            except Exception as exc:
                record["error"] = f"GenerationError: {exc}"

        records.append(record)

    # 4. Compute Aggregate Statistics
    num_eval = len(records)
    agg_retrieval: Dict[str, float] = {}
    for k in (1, 3, 5, 10):
        for metric_name in ("hit", "recall", "both"):
            key = f"{metric_name}_{k}"
            vals = [r["retrieval_metrics"].get(key, 0.0) for r in records if "retrieval_metrics" in r]
            agg_retrieval[key] = round(sum(vals) / max(len(vals), 1), 4)

    agg_generation: Dict[str, float] = {}
    if not dry_run:
        em_vals = [r["generation_metrics"].get("exact_match", 0.0) for r in records if "generation_metrics" in r]
        f1_vals = [r["generation_metrics"].get("f1", 0.0) for r in records if "generation_metrics" in r]
        agg_generation["exact_match"] = round(sum(em_vals) / max(len(em_vals), 1), 4)
        agg_generation["f1"] = round(sum(f1_vals) / max(len(f1_vals), 1), 4)

    summary: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pipeline": "vanilla_rag_baseline",
            "pilot_flag": True,
            "disclaimer": (
                "PILOT REPRODUCTION ONLY: Evaluated on train[:100] for legacy comparison. "
                "The index was built from contexts in train[:1000], so context overlap exists. "
                "This does not establish generalization to unseen queries."
            ),
            "split": split_str,
            "num_questions": num_eval,
            "embedding_model": MODEL_NAME,
            "collection_name": COLLECTION_NAME,
            "retrieval_limit": TOP_K_RETRIEVAL,
            "context_limit": TOP_K_GENERATION,
            "llm_model": llm_model_name,
            "llm_base_url": LLAMA_BASE_URL,
            "generation_params": {"temperature": 0.0, "max_tokens": 256},
            "runtime_seconds": {
                "total_retrieval": round(total_retrieval_time, 2),
                "total_generation": round(total_generation_time, 2),
                "avg_retrieval_per_q": round(total_retrieval_time / max(num_eval, 1), 4),
                "avg_generation_per_q": round(total_generation_time / max(num_eval, 1), 4),
            }
        },
        "aggregate_metrics": {
            "retrieval": agg_retrieval,
            "generation": agg_generation
        },
        "records": records
    }

    # 5. Write to output file without overwriting
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_path = OUTPUT_DIR / output_filename
    if target_path.exists():
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = OUTPUT_DIR / f"baseline_pilot_100_{timestamp_str}.json"

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=======================================================")
    print(f"BASELINE PILOT EVALUATION COMPLETE")
    print(f"=======================================================")
    print(f"Results written to: {target_path}")
    print(f"Aggregate Retrieval Metrics:")
    for k in (1, 3, 5, 10):
        print(f"  Hit@{k}:    {agg_retrieval.get(f'hit_{k}', 0.0):.4f} | "
              f"Recall@{k}: {agg_retrieval.get(f'recall_{k}', 0.0):.4f} | "
              f"Both@{k}:   {agg_retrieval.get(f'both_{k}', 0.0):.4f}")
    if not dry_run:
        print(f"Aggregate Generation Metrics:")
        print(f"  Exact Match (EM): {agg_generation.get('exact_match', 0.0):.4f}")
        print(f"  Token F1:         {agg_generation.get('f1', 0.0):.4f}")
        print(f"Average Timing: Retrieval={summary['metadata']['runtime_seconds']['avg_retrieval_per_q']}s | "
              f"Generation={summary['metadata']['runtime_seconds']['avg_generation_per_q']}s")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vanilla RAG baseline evaluation harness.")
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS, help="Number of questions (default: 100)")
    parser.add_argument("--dry-run", action="store_true", help="Run retrieval only, skipping LLM generation")
    parser.add_argument("--output", type=str, default="baseline_pilot_100.json", help="Output JSON filename")
    args = parser.parse_args()

    run_baseline_evaluation(
        num_questions=args.num_questions,
        split_str=f"train[:{args.num_questions}]",
        output_filename=args.output,
        dry_run=args.dry_run
    )
