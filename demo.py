#!/usr/bin/env python3
"""End-to-End RAG Verification Interactive & Scriptable Demo.

Demonstrates the complete pipeline from user question to evidence-grounded final answer:
1. Question input
2. Passage retrieval from Qdrant
3. Draft answer generation
4. Claim extraction & triage scoring
5. Policy-guided sequential verification (LinUCB / CTS / FIFO / Exhaustive)
6. Evidence-grounded final answer synthesis
7. Full cost, citation, and limitation disclosure
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

# Ensure project root is importable and stdout is UTF-8 on Windows
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import RAGVerificationPipeline, PipelineResult
from src.synthesis.citations import check_citations, claim_evidence_map


DEFAULT_QUESTION = "What State has a Disney Resort & Spa that is a beachside hotel, resort and vacation destination offering complimentary children's activities and programs and that Djuan Rivers was a General Manager at?"


def format_currency(tokens: int) -> str:
    """Format token count for display."""
    return f"{tokens:,} tokens"


def print_banner() -> None:
    print("=" * 80)
    print("      EVIDENCE-GROUNDED RAG SEQUENTIAL VERIFICATION SYSTEM — DEMO")
    print("=" * 80)
    print("Pipeline: Retrieval -> Draft Gen -> Claim Triage -> Policy Controller -> Grounded Synthesis")
    print("Live mode needs Qdrant (:6333) and llama-server (LLAMA_SERVER_URL, default :8080). --dry-run uses MOCKS.")
    print("Default policy: ContextualMAB_LinUCB, frozen warm-start checkpoint (trained on train[100:300]; no online learning here)")
    print("=" * 80)


def _is_mock(obj) -> bool:
    return type(obj).__name__.startswith("Mock")


def print_components(pipeline: RAGVerificationPipeline, policy: str, draft_supplied: bool = False) -> None:
    """State exactly which components are real and which are mocks for this run."""
    print("\nCOMPONENTS FOR THIS RUN")
    print("-" * 60)
    draft_mock = pipeline.dry_run or _is_mock(pipeline.synthesizer)
    rows = [
        ("Retriever", pipeline.retriever, _is_mock(pipeline.retriever)),
    ]
    rows.append(("Draft generator", None if draft_supplied else pipeline.synthesizer, draft_mock))
    rows += [
        ("Assessor", pipeline.assessor, _is_mock(pipeline.assessor)),
        ("Repairer", pipeline.repairer, _is_mock(pipeline.repairer)),
        ("Synthesizer", pipeline.synthesizer, _is_mock(pipeline.synthesizer)),
    ]
    for label, obj, mock in rows:
        if obj is None:
            print(f"  {label:<16} SKIPPED  initial generation not run: draft supplied by the user (--draft)")
            continue
        extra =f" (collection: {obj.collection_name})" if label == "Retriever" and hasattr(obj, "collection_name") else ""
        print(f"  {label:<16} {'MOCK' if mock else 'REAL':<5} {type(obj).__name__}{extra}")
    print(f"  {'Decomposer':<16} RULE  {type(pipeline.decomposer).__name__} (heuristic, no LLM)")
    print(f"  {'CTS triage':<16} RULE  {type(pipeline.triage_scorer).__name__} (heuristic, no LLM)")
    print(f"  {'Policy':<16} REAL  {policy}")


def print_trace(trace) -> None:
    """Print each controller step: arm, rationale, evidence, verdict, repair, re-check, budget."""
    print("Execution Trace:")
    for step in trace.steps:
        if step.action.value == "STOP":
            continue
        cts = f"{step.triage_score:.2f}" if step.triage_score is not None else "N/A"
        print(f"\n  Step {step.step_number}: {step.action.value} on {step.claim_id} "
              f"(CTS={cts}; {step.previous_state} -> {step.new_state})")
        print(f"    claim:     {step.claim_text}")
        print(f"    selected:  {step.selection_rationale}")
        for p in step.retrieved_passages[:3]:
            print(f"    evidence:  [{p['evidence_id']}] {p['doc_title']}")
        if step.assessment_label:
            print(f"    verdict:   {step.assessment_label} (conf {step.assessment_confidence:.2f})")
        if step.repair_data:
            rd = step.repair_data
            print(f"    repair:    \"{rd.get('proposed_revision')}\"  evidence={rd.get('supporting_evidence_ids')}")
            rv = rd.get("reverification")
            if rv:
                rc = rv.get("cost") or {}
                print(f"    re-verify: {rv['label']} (conf {rv['confidence']:.2f}) -> "
                      f"{'PASSED' if rv['verified'] else 'NOT VERIFIED'}  "
                      f"[cost: calls={rc.get('model_calls', 0)} tokens={rc.get('total_tokens', 0)}; charged to the budget, "
                      f"which also used 1 step for it, so steps_taken exceeds the number of trace rows by 1 per re-verification]")
            else:
                print("    re-verify: not attempted")
        print(f"    outcome:   {step.action_outcome or 'Done'}" + (f" [ERROR: {step.error}]" if step.error else ""))
        cost = step.action_cost or {}
        print(f"    cost:      calls={cost.get('model_calls', 0)} tokens={cost.get('total_tokens', 0)} "
              f"| remaining budget: {step.remaining_budget}")


def preflight(dry_run: bool) -> None:
    """Fail fast with a clear message if live services are down; never fall back to mocks."""
    if dry_run:
        return
    llm = urlparse(os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1"))
    targets = [("Qdrant", "127.0.0.1", 6333), ("llama-server", llm.hostname or "127.0.0.1", llm.port or 8080)]
    down = []
    for name, host, port in targets:
        try:
            socket.create_connection((host, port), timeout=2).close()
        except OSError:
            down.append(f"{name} ({host}:{port})")
    if down:
        sys.exit("ERROR: live service(s) not reachable: " + ", ".join(down)
                 + ".\nStart them, or pass --dry-run for the clearly-labelled MOCK mode.")


def collect_evidence(result: PipelineResult) -> list:
    """Every passage the run actually retrieved: initial retrieval plus per-claim retrieval in the trace."""
    seen, out = set(), []
    pools = [result.evidence_passages]
    if result.trace:
        pools += [s.retrieved_passages for s in result.trace.steps]
    for pool in pools:
        for p in pool:
            if p.get("evidence_id") and p["evidence_id"] not in seen:
                seen.add(p["evidence_id"])
                out.append(p)
    return out


def render_result(result: PipelineResult, verbose: bool = True) -> None:
    print("\n" + "=" * 80)
    print(f"QUESTION: {result.question}")
    print(f"VERIFICATION POLICY: {result.policy_name}")
    print("=" * 80)

    print("\n[1] INITIAL RETRIEVAL & DRAFT ANSWER")
    print("-" * 60)
    if result.draft_source == "user_supplied":
        print("DRAFT SOURCE: USER-SUPPLIED (--draft). Initial draft generation was SKIPPED; no LLM wrote this text.")
    print(f"Initial Draft Answer:\n  \"{result.initial_draft}\"")
    print("\nInitial Retrieved Passages:")
    for i, p in enumerate(result.evidence_passages[:3]):
        score_str = f"(score: {p['retrieval_score']:.4f})" if p.get('retrieval_score') else ""
        print(f"  [{i+1}] {p['doc_title']} {score_str}  id={p['evidence_id']}")
        print(f"      {p['snippet']}")

    print("\n[2] CLAIM EXTRACTION & TRIAGE")
    print("-" * 60)
    print(f"{'Claim ID':<10} {'Triage':<8} {'State':<14} {'Assessor Verdict':<20} {'Claim Text'}")
    print("-" * 75)
    for c in result.claims:
        triage = f"{c['triage_score']:.2f}" if c.get('triage_score') is not None else "N/A"
        verdict = f"{c['assessment_label']} (conf: {c['confidence']:.2f})" if c.get('assessment_label') else "NOT EVALUATED"
        state = c['state']
        text = c['text'][:35] + ("..." if len(c['text']) > 35 else "")
        print(f"{c['claim_id']:<10} {triage:<8} {state:<14} {verdict:<20} {text}")

    print("\n[3] SEQUENTIAL CONTROLLER EXECUTION")
    print("-" * 60)
    if result.budget:
        print(f"Budget: {result.budget['max_steps']} steps, {result.budget['max_model_calls']} model calls "
              f"(per claim: {result.budget['max_retrievals_per_claim']} retrievals / "
              f"{result.budget['max_verifications_per_claim']} verifications / {result.budget['max_repairs_per_claim']} repairs)")
    print(f"Steps Taken: {result.steps_taken} | Stopping Reason: {result.stopping_reason}")
    if result.trace and verbose:
        print_trace(result.trace)

    print("\n[4] FINAL GROUNDED ANSWER SYNTHESIS")
    print("-" * 60)
    print(f"Final Answer:\n  \"{result.final_answer}\"")
    print(f"\nAnswer Changed from Draft: {'YES (Refined/Grounded)' if result.answer_changed else 'NO (Draft Preserved)'}")
    print(f"Grounded in Evidence:      {'YES' if result.is_grounded else 'NO (Insufficient Evidence)'}")

    if result.synthesis and result.synthesis.get("cited_sources"):
        print(f"Cited Sources: {', '.join(result.synthesis['cited_sources'])}")

    cc = check_citations(result.final_answer, collect_evidence(result))
    print("\nCITATION CHECK (inline [Title] markers vs. passages retrieved in this run)")
    print(f"  markers in final answer: {cc['n_markers']} | resolved: {len(cc['resolved'])} | unresolved: {len(cc['unresolved'])}")
    for mk, ids in cc["resolved"].items():
        print(f"    [{mk}] -> {', '.join(ids)}")
    for mk in cc["unresolved"]:
        print(f"    [{mk}] -> NOT FOUND in retrieved evidence")
    for note in (result.synthesis or {}).get("uncertainty_notes", []):
        if note.startswith("Removed"):
            print(f"  note: {note}")

    av = result.answer_validation or {}
    if av.get("enforced"):
        source = {"llm_accepted": "LLM rewrite accepted after validation",
                  "llm_rejected_stitched": "LLM rewrite REJECTED by validation; deterministic stitched answer used",
                  "stitched_fallback": "LLM unavailable; deterministic stitched answer used",
                  "validation_failed_abstention": "validation failed; safe abstention"}.get(av.get("polish_status"), av.get("polish_status"))
        print(f"\nANSWER VALIDATION (enforced): {'PASSED' if not av['violations'] else 'FAILED ' + str(av['violations'][:2])}"
              f" | answer source: {source}")
        print("  checks: only whole verified claims | no content of unresolved/unverified-repair claims | no repeated sentence | "
              "no verified claim dropped | each line cites exactly its claims' supporting sources")
        for cc in (result.synthesis or {}).get("claim_citations", []):
            tgt = f"[{cc['title']}] {cc['evidence_ids']}" if cc["title"] else "no supporting passage (uncited)"
            print(f"  {cc['claim_id']:<8} -> {tgt}  overlap={cc['overlap']:.2f}")

    unbacked = [r for r in result.answer_audit if not r["backed"]]
    print(f"\nANSWER AUDIT (lexical, report-only; the enforced check is above):{len(result.answer_audit) - len(unbacked)} of {len(result.answer_audit)} "
          "final-answer sentences are backed by a verified claim")
    for r in unbacked:
        print(f"  NOT BACKED (overlap {r['best_verified_claim_overlap']:.2f}): {r['sentence'][:150]}")

    print("\nCLAIM -> EVIDENCE MAP (deterministic content-word overlap against the retrieved passages; computed by the pipeline, not by an LLM)")
    for row in claim_evidence_map(result.claims, collect_evidence(result)):
        tgt = f"{row['best_evidence_id']} [{row['best_doc_title']}]" if row["best_evidence_id"] else "no overlapping passage"
        print(f"  {row['claim_id']:<8} {row['state']:<10} overlap={row['content_word_overlap']:.2f} -> {tgt}")

    if result.warnings:
        print("\n[5] UNCERTAINTY DISCLOSURES & LIMITATIONS")
        print("-" * 60)
        for w in result.warnings:
            print(f"  ! {w}")

    print("\n[6] RESOURCE COST ACCOUNTING")
    print("-" * 60)
    c_init = result.initial_cost
    c_verif = result.verification_cost
    c_synth = result.synthesis_cost
    c_tot = result.total_cost

    print(f"{'Phase':<22} {'Calls':<8} {'Prompt Tok':<12} {'Compl Tok':<12} {'Total Tok':<12} {'Latency'}")
    print("-" * 75)
    print(f"{'Initial (Ret+Draft)':<22} {c_init.get('model_calls', 0):<8} {c_init.get('prompt_tokens', 0):<12} {c_init.get('completion_tokens', 0):<12} {c_init.get('total_tokens', 0):<12} {c_init.get('latency_seconds', 0.0):.2f}s")
    print(f"{'Verification Loop':<22} {c_verif.get('model_calls', 0):<8} {c_verif.get('prompt_tokens', 0):<12} {c_verif.get('completion_tokens', 0):<12} {c_verif.get('total_tokens', 0):<12} {c_verif.get('latency_seconds', 0.0):.2f}s")
    print(f"{'Grounded Synthesis':<22} {c_synth.get('model_calls', 0):<8} {c_synth.get('prompt_tokens', 0):<12} {c_synth.get('completion_tokens', 0):<12} {c_synth.get('total_tokens', 0):<12} {c_synth.get('latency_seconds', 0.0):.2f}s")
    print("-" * 75)
    print(f"{'TOTAL END-TO-END':<22} {c_tot.get('model_calls', 0):<8} {c_tot.get('prompt_tokens', 0):<12} {c_tot.get('completion_tokens', 0):<12} {c_tot.get('total_tokens', 0):<12} {c_tot.get('latency_seconds', 0.0):.2f}s")
    rep_cls = getattr(result, "repairer_class", "") or "Unknown"
    print(f"\nDISCLOSURE: The repair action uses {rep_cls}.")
    print("=" * 80 + "\n")


def run_demo(
    question: str,
    policy: str = "ContextualMAB_LinUCB",
    dry_run: bool = False,
    output_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    collection: Optional[str] = None,
    long_answer: bool = False,
    top_k: int = 5,
    draft: Optional[str] = None,
    max_steps: Optional[int] = None,
    max_calls: Optional[int] = None,
) -> PipelineResult:
    preflight(dry_run)
    pipeline = RAGVerificationPipeline(dry_run=dry_run, collection_name=collection, long_answer=long_answer)
    print_components(pipeline, policy, draft_supplied=draft is not None)
    result = pipeline.run(
        question=question,
        policy_name=policy,
        checkpoint_path=checkpoint,
        top_k=top_k,
        draft_answer=draft,
        max_steps=max_steps,
        max_model_calls=max_calls,
    )
    render_result(result)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            payload = result.to_dict()
            payload["run_settings"] = {"collection": collection or "default (hotpotqa)", "long_answer": long_answer,
                                       "top_k": top_k, "dry_run": dry_run,
                                       "draft_source": result.draft_source, "budget": result.budget}
            payload["citation_check"] = check_citations(result.final_answer, collect_evidence(result))
            payload["claim_evidence_map"] = claim_evidence_map(result.claims, collect_evidence(result))
            json.dump(payload, f, indent=2)
        print(f"Result saved to: {out_p}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Evidence-Grounded RAG Verification Demo")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="User natural language question (or use default if omitted)")
    parser.add_argument("--policy", "-p", type=str, default="ContextualMAB_LinUCB",
                        choices=["ContextualMAB_LinUCB", "DeterministicCTS", "StaticCTSRanked", "StaticFIFO", "ExhaustiveReference", "VanillaRAG"],
                        help="Verification budget allocation policy (default: ContextualMAB_LinUCB)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run in offline mock mode without connecting to live Qdrant or llama-server")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LinUCB warm-start checkpoint (default: experiments/runs/linucb_warmstart_train100_300.json)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to save output JSON result")
    parser.add_argument("--collection", type=str, default=None,
                        help="Qdrant collection to retrieve from (default: the HotpotQA collection). E.g. alce_asqa_bge-large-en-v1.5")
    parser.add_argument("--long-answer", action="store_true",
                        help="Draft a multi-sentence answer (4-6 sentences) so several atomic claims are produced")
    parser.add_argument("--top-k", type=int, default=5, help="Passages retrieved for the draft (default 5)")
    parser.add_argument("--draft", type=str, default=None,
                        help="Supply the draft answer yourself. Initial LLM draft generation is SKIPPED (and labelled as such); "
                             "retrieval, claim extraction, CTS, MAB, assessment, repair, re-verification and synthesis all run on your text")
    parser.add_argument("--max-steps", type=int, default=None, help="Override the global step budget (default 5)")
    parser.add_argument("--max-calls", type=int, default=None, help="Override the global model-call budget (default 8)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive CLI mode to enter multiple questions")

    args = parser.parse_args()
    print_banner()

    if args.interactive:
        print("Entering interactive mode. Press Ctrl+C or type 'exit' to quit.\n")
        while True:
            try:
                q = input("\nEnter your question [or press Enter for default]: ").strip()
                if q.lower() in ("exit", "quit", "q"):
                    break
                question_text = q if q else DEFAULT_QUESTION
                run_demo(question_text, policy=args.policy, dry_run=args.dry_run, checkpoint=args.checkpoint,
                         collection=args.collection, long_answer=args.long_answer, top_k=args.top_k)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive demo.")
                break
    else:
        question_text = args.question if args.question else DEFAULT_QUESTION
        run_demo(
            question=question_text,
            policy=args.policy,
            dry_run=args.dry_run,
            output_path=args.output,
            checkpoint=args.checkpoint,
            collection=args.collection,
            long_answer=args.long_answer,
            top_k=args.top_k,
            draft=args.draft,
            max_steps=args.max_steps,
            max_calls=args.max_calls,
        )


if __name__ == "__main__":
    main()
