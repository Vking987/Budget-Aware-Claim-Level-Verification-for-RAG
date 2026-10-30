#!/usr/bin/env python3
"""Execute 5-question live demonstration suite and save timestamped results."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import RAGVerificationPipeline, DEFAULT_WARMSTART_CHECKPOINT


REPRESENTATIVE_QUESTIONS = [
    {
        "id": "q1_disney_state",
        "question": "What State has a Disney Resort & Spa that is a beachside hotel, resort and vacation destination offering complimentary children's activities and programs and that Djuan Rivers was a General Manager at?",
        "gold_expected": "Hawaii",
    },
    {
        "id": "q2_airline_passengers",
        "question": "The airline operating in Whitehorse, Yukon handled how many passengers in 2012?",
        "gold_expected": "294,000",
    },
    {
        "id": "q3_china_cities",
        "question": "Are Wenling and Xinzheng both in China?",
        "gold_expected": "yes",
    },
    {
        "id": "q4_coach_birth_date",
        "question": "What is the birth date of the coach of the 2014-15 Michigan Wolverines men's basketball team?",
        "gold_expected": "February 5, 1953",
    },
    {
        "id": "q5_monarchs_kingdom",
        "question": "Prince William, Duke of Gloucester and Prince George of Denmark, were eventually monarchs of which Kingdom?",
        "gold_expected": "Great Britain",
    },
]


def run_live_suite():
    print("=" * 80)
    print("  EXECUTING LIVE 5-QUESTION DEMO SUITE (REAL QDRANT + LLAMA-SERVER)")
    print("=" * 80)
    print(f"LinUCB Checkpoint: {DEFAULT_WARMSTART_CHECKPOINT}")
    assert DEFAULT_WARMSTART_CHECKPOINT.exists(), f"Missing checkpoint {DEFAULT_WARMSTART_CHECKPOINT}"

    pipeline = RAGVerificationPipeline(dry_run=False)

    results_data = []
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "experiments" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"live_demo_{timestamp_str}.json"

    suite_start = time.monotonic()

    for idx, item in enumerate(REPRESENTATIVE_QUESTIONS):
        qid = item["id"]
        q_text = item["question"]
        gold = item.get("gold_expected", "")

        print(f"\n[{idx+1}/5] Running: {q_text[:70]}...")
        q_start = time.monotonic()

        res = pipeline.run(
            question=q_text,
            policy_name="ContextualMAB_LinUCB",
            checkpoint_path=DEFAULT_WARMSTART_CHECKPOINT,
        )
        elapsed = time.monotonic() - q_start

        verified_cnt = sum(1 for c in res.claims if c["state"] == "VERIFIED")
        unresolved_cnt = sum(1 for c in res.claims if c["state"] in ("UNRESOLVED", "PENDING", "DEFERRED"))

        print(f"  Draft Answer:  \"{res.initial_draft}\"")
        print(f"  Final Answer:  \"{res.final_answer}\"")
        print(f"  Answer Changed: {'YES' if res.answer_changed else 'NO'} | Grounded: {'YES' if res.is_grounded else 'NO'}")
        print(f"  Claims (V:{verified_cnt}/U:{unresolved_cnt}) | Steps: {res.steps_taken} | Time: {elapsed:.1f}s")
        if res.warnings:
            for w in res.warnings:
                print(f"  ! {w}")

        entry = {
            "index": idx,
            "id": qid,
            "question": q_text,
            "expected_target": gold,
            "initial_draft": res.initial_draft,
            "final_synthesized_answer": res.final_answer,
            "answer_changed": res.answer_changed,
            "is_grounded": res.is_grounded,
            "has_unresolved_limitations": res.has_unresolved_limitations,
            "policy_used": res.policy_name,
            "steps_taken": res.steps_taken,
            "stopping_reason": res.stopping_reason,
            "claims": res.claims,
            "evidence_passages": res.evidence_passages,
            "synthesis": res.synthesis,
            "warnings": res.warnings,
            "cost": {
                "initial": res.initial_cost,
                "verification": res.verification_cost,
                "synthesis": res.synthesis_cost,
                "total": res.total_cost,
            },
            "latency_seconds": round(elapsed, 2),
        }
        results_data.append(entry)

    total_wall = time.monotonic() - suite_start
    print("\n" + "=" * 80)
    print(f"LIVE DEMO SUITE COMPLETE in {total_wall:.1f}s")
    print("=" * 80)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_used": str(DEFAULT_WARMSTART_CHECKPOINT),
        "total_questions": len(results_data),
        "total_wall_seconds": round(total_wall, 2),
        "questions_where_answer_changed": sum(1 for r in results_data if r["answer_changed"]),
        "questions_grounded": sum(1 for r in results_data if r["is_grounded"]),
        "records": results_data,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved live demo output artifact: {out_file}")


if __name__ == "__main__":
    run_live_suite()
