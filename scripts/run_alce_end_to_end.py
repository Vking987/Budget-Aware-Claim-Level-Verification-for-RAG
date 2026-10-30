#!/usr/bin/env python3
"""End-to-end ALCE-ASQA evaluation of claim-verification budget policies (real retrieval, generation, decomposition, verification, repair).

Per question (identical for every policy): real Qdrant retrieval of 5 passages -> real LLM long-form draft -> rule-based claim decomposition + CTS ->
VerificationController under a call budget with the policy under test (real LlamaAssessor / LLMRepairer, verdicts memoised at temperature 0 so every
unique claim is verified by the model once and replayed, with its recorded cost, for other policies) -> final answer STITCHED deterministically from
the verified claims (the optional LLM rewrite is skipped in this benchmark). Stages: `--stage prep` (retrieval + draft, cached), `--stage run`.

Metrics: ALCE STR-EM (reimplemented; ALCE's eval.py, exact_presence + compute_str_em, was read, but its normalize_answer was not visible, so
SQuAD-style normalisation is ASSUMED). Citations are stripped before scoring. Loss decomposition: retrieval (1 - STR-EM of the retrieved passages),
generation (retrieved - draft), selection/verification (draft - final), repair gain (final - final without repaired claims).
Verifier errors use a PROXY, not gold: a claim is 'context-ungrounded' if no retrieved passage supports it under the lexical support rule.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import string
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import policy_controls as PC  # noqa: E402
import run_ragtruth_comparison as R  # noqa: E402
from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus  # noqa: E402
from src.controller.baselines import RandomPolicy  # noqa: E402
from src.controller.controller import VerificationController  # noqa: E402
from src.controller.models import ControllerAction, PerClaimBudget, VerificationBudget  # noqa: E402
from src.evidence.models import AssessmentLabel, EvidencePassage  # noqa: E402
from src.synthesis.grounding import build_items, stitch_answer, support_for_claim  # noqa: E402

FLAG = (AssessmentLabel.CONTRADICTED.value, AssessmentLabel.CONFLICTED.value)
_CITATION = re.compile(r"\[[^\[\]\n]{1,200}\]")


# ----------------------------------------------------------------------------- ALCE-style metric
def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_presence(short_answers: List[str], text: str) -> bool:
    n_text = normalize_answer(text)
    return any(normalize_answer(a) in n_text for a in short_answers if normalize_answer(a))


def str_em(qa_pairs: List[Dict[str, Any]], text: str) -> float:
    """Per-question STR-EM: mean over disambiguated questions of 'some short answer appears in the text'."""
    if not qa_pairs:
        return 0.0
    return float(np.mean([exact_presence(p["short_answers"], text) for p in qa_pairs]))


def strip_citations(text: str) -> str:
    return " ".join(_CITATION.sub(" ", text or "").split())


def save_cache(cache, tries: int = 8) -> None:
    """cache.save() with retries: on Windows os.replace can fail transiently (AccessDenied) while a scanner holds the file.
    A failed save only costs re-computation on a later resume, so it is never fatal."""
    for k in range(tries):
        try:
            cache.save()
            return
        except PermissionError:
            time.sleep(1.5 * (k + 1))
    print("WARNING: cache save failed repeatedly; continuing without persisting this checkpoint", flush=True)


# ----------------------------------------------------------------------------- stage A: retrieval + draft (cached)
def passage_to_dict(p: EvidencePassage) -> Dict[str, Any]:
    return {"evidence_id": p.evidence_id, "doc_title": p.doc_title, "source_text": p.source_text, "retrieval_score": p.retrieval_score}


def claim_to_dict(c: ClaimRecord) -> Dict[str, Any]:
    return {"claim_id": c.claim_id, "source_sentence_id": c.source_sentence_id, "original_text": c.original_text,
            "normalized_text": c.normalized_text, "claim_type": c.claim_type.value,
            "decomposition_status": c.decomposition_status.value, "triage_score": c.triage_score}


def claim_from_dict(d: Dict[str, Any]) -> ClaimRecord:
    return ClaimRecord(claim_id=d["claim_id"], source_sentence_id=d["source_sentence_id"], original_text=d["original_text"],
                       normalized_text=d["normalized_text"], claim_type=ClaimType(d["claim_type"]),
                       decomposition_status=DecompositionStatus(d["decomposition_status"]), state=ClaimState.PENDING,
                       triage_score=d["triage_score"])


def prepare_question(pipeline, q: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """Real retrieval, real draft generation, rule-based decomposition + CTS, using the pipeline's own components."""
    question = q["question"]
    dummy = ClaimRecord(claim_id="init_query", source_sentence_id="s0", original_text=question, normalized_text=question,
                        claim_type=ClaimType.FACTUAL_RELATION, decomposition_status=DecompositionStatus.ATOMIC)
    passages, ret_cost = pipeline.retriever.retrieve(dummy, top_k=top_k)
    draft, gen_cost = pipeline._generate_draft(question, passages)
    claims: List[ClaimRecord] = []
    for sent in pipeline.content_filter.process(draft):
        if sent.is_content_bearing:
            claims.extend(pipeline.decomposer.decompose(sent))
    for c in claims:
        c.triage_score = pipeline.triage_scorer.score_claim(c, context_text=draft)
    return {"sample_id": str(q["sample_id"]), "question": question, "qa_pairs": q["qa_pairs"],
            "passages": [passage_to_dict(p) for p in passages], "draft": draft, "claims": [claim_to_dict(c) for c in claims],
            "prep_cost": {"retrieval_calls": ret_cost.model_calls, "generation_calls": gen_cost.model_calls,
                          "prompt_tokens": (ret_cost.prompt_tokens or 0) + (gen_cost.prompt_tokens or 0),
                          "completion_tokens": gen_cost.completion_tokens or 0,
                          "latency_seconds": (ret_cost.latency_seconds or 0.0) + (gen_cost.latency_seconds or 0.0)}}


# ----------------------------------------------------------------------------- stage B: one controller run
def run_one(prep: Dict[str, Any], policy, budget: int, retriever, assessor, repairer, run_id: str) -> Dict[str, Any]:
    claims = [claim_from_dict(c) for c in prep["claims"]]
    passages = [EvidencePassage(evidence_id=p["evidence_id"], doc_title=p["doc_title"], source_text=p["source_text"],
                                retrieval_score=p["retrieval_score"]) for p in prep["passages"]]
    store = {c.claim_id: list(passages) for c in claims}
    ctl = VerificationController(retriever=retriever, assessor=assessor, repairer=repairer, policy=policy,
                                 global_budget=VerificationBudget(max_steps=50, max_model_calls=budget), per_claim_budget=PerClaimBudget())
    trace = ctl.run(claims=claims, initial_evidence={k: list(v) for k, v in store.items()}, run_id=run_id)

    first_label, repairs = {}, {}
    n_repair = n_reverified = 0
    for s in trace.steps:
        if s.action == ControllerAction.VERIFY and s.assessment_label and not s.error and s.claim_id not in first_label:
            first_label[s.claim_id] = s.assessment_label
        if s.action == ControllerAction.REPAIR and s.repair_data:
            n_repair += 1
            rv = s.repair_data.get("reverification") or {}
            if rv.get("verified"):
                n_reverified += 1
                repairs[s.claim_id] = s.repair_data
    verified_ids = {c.claim_id for c in claims if c.state == ClaimState.VERIFIED}
    items_all = build_items(claims, repairs, store, verified_ids | set(repairs))
    items_no_repair = build_items(claims, {}, store, verified_ids)
    answer, answer_no_repair = stitch_answer(items_all), stitch_answer(items_no_repair)
    ungrounded = {c.claim_id: support_for_claim(c.normalized_text, passages)["title"] is None for c in claims}
    tc = trace.total_cost if isinstance(trace.total_cost, dict) else trace.total_cost.to_dict()
    qa = prep["qa_pairs"]
    in_final = {c.claim_id for c in claims if c.claim_id in verified_ids or c.claim_id in repairs}
    return {
        "sample_id": prep["sample_id"], "n_claims": len(claims), "n_verified": len(verified_ids), "n_verified_claims_in_answer": len(in_final),
        "n_flagged": sum(1 for v in first_label.values() if v in FLAG), "n_checked": len(first_label),
        "n_repair": n_repair, "n_reverified_pass": n_reverified,
        "str_em_final": str_em(qa, strip_citations(answer)), "str_em_final_no_repair": str_em(qa, strip_citations(answer_no_repair)),
        "ungrounded_total": sum(ungrounded.values()),
        "ungrounded_checked": sum(1 for k, u in ungrounded.items() if u and k in first_label),
        "ungrounded_flagged": sum(1 for k, u in ungrounded.items() if u and first_label.get(k) in FLAG),
        "grounded_checked": sum(1 for k, u in ungrounded.items() if (not u) and k in first_label),
        "grounded_flagged": sum(1 for k, u in ungrounded.items() if (not u) and first_label.get(k) in FLAG),
        # what the answer actually says: verified claims by their original text, repaired claims by their REVISION
        "ungrounded_in_final": sum(1 for k, u in ungrounded.items() if u and k in verified_ids)
        + sum(1 for k, rd in repairs.items()
              if support_for_claim(rd.get("proposed_revision") or "", passages)["title"] is None),
        "model_calls": tc.get("model_calls") or 0, "total_tokens": tc.get("total_tokens") or 0, "latency_seconds": tc.get("latency_seconds") or 0.0,
        "steps": trace.total_steps, "final_answer": answer,
    }


def question_baselines(prep: Dict[str, Any]) -> Dict[str, float]:
    ctx = " ".join(p["source_text"] for p in prep["passages"])
    return {"retrieval_str_em": str_em(prep["qa_pairs"], ctx), "draft_str_em": str_em(prep["qa_pairs"], strip_citations(prep["draft"]))}


# ----------------------------------------------------------------------------- aggregation
def boot_ci(per_q: np.ndarray, n_boot: int = 2000, seed: int = 0):
    if len(per_q) < 5:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_q), size=(n_boot, len(per_q)))
    v = per_q[idx].mean(axis=1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def aggregate(runs: Dict[tuple, List[Dict[str, Any]]], bases: List[Dict[str, float]], qids: List[str], metric: str = "str_em_final"):
    """runs[(budget, kind)] = list over seeds of per-question dict rows; returns per-question arrays averaged over seeds."""
    out = {}
    for key, seed_rows in runs.items():
        mat = np.array([[next(r for r in rows if r["sample_id"] == q)[metric] for q in qids] for rows in seed_rows])
        out[key] = mat.mean(axis=0)
    return out


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asqa-file", default=str(ROOT / "data/alce/raw/ALCE-data/asqa_eval_gtr_top100.json"))
    ap.add_argument("--collection", default="alce_asqa_eval_bge-large-en-v1.5")
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--n-questions", type=int, default=30)
    ap.add_argument("--stage", choices=["prep", "run", "all"], default="all")
    ap.add_argument("--budgets", default="2,4,8")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--ablation-ts", default="20261029_002617")
    ap.add_argument("--train-responses", type=int, default=1000)
    ap.add_argument("--verifier", choices=["llama", "nli"], default="llama",
                    help="llama = the 1B LLM assessor of the first run; nli = DeBERTa NLI cross-encoder (repair and drafts stay on the 1B model)")
    ap.add_argument("--cache", default=None, help="verdict cache (default depends on --verifier; NEVER share one cache across verifiers: keys omit the assessor)")
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/runs"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    if args.cache is None:
        args.cache = str(ROOT / "experiments/cache" / ("alce_e2e_llm_cache.pkl" if args.verifier == "llama" else "alce_e2e_nli_cache.pkl"))

    from src.pipeline import RAGVerificationPipeline
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    questions = json.load(open(args.asqa_file, encoding="utf-8"))[args.start: args.start + args.n_questions]
    R.preflight_llm()
    from qdrant_client import QdrantClient
    from src.controller.retriever import QdrantRetriever
    # generous client timeout: the default (5 s) fails when Qdrant is busy; this changes robustness, not what is measured
    retriever = QdrantRetriever(client=QdrantClient(host="localhost", port=6333, timeout=120), collection_name=args.collection)
    pipe = RAGVerificationPipeline(retriever=retriever, collection_name=args.collection, long_answer=True)
    prep_path = out_dir / f"alce_e2e_prep_{args.collection}_{args.start}_{args.n_questions}.json"
    prep_cache: Dict[str, Any] = json.load(open(prep_path, encoding="utf-8")) if prep_path.exists() else {}
    t0 = time.monotonic()
    for i, q in enumerate(questions):
        sid = str(q["sample_id"])
        if sid not in prep_cache:
            prep_cache[sid] = prepare_question(pipe, q, args.top_k)
            prep_path.write_text(json.dumps(prep_cache), encoding="utf-8")
            print(f"[prep {i + 1}/{len(questions)}] {len(prep_cache[sid]['claims'])} claims ({time.monotonic() - t0:.0f}s) {q['question'][:60]}", flush=True)
    if args.stage == "prep":
        return {"prep_path": str(prep_path), "n": len(prep_cache)}
    preps = [prep_cache[str(q["sample_id"])] for q in questions]
    qids = [p["sample_id"] for p in preps]

    # ---- policies: fitted ONLY on RAGTruth TRAIN claims (oracle control), frozen, then transferred to ASQA
    from src.controller.mab_policy import ContextualMABPolicy
    from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter
    seeds = [int(s) for s in args.seeds.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    tr = RAGTruthDatasetAdapter()
    tr.load_from_jsonl(R.DATA_DIR / "source_info.jsonl", R.DATA_DIR / "response.jsonl", split="train")
    cache = R.LLMCache(Path(args.cache))
    if args.verifier == "nli":
        from src.evidence.nli_assessor import NLIAssessor
        inner_assessor = NLIAssessor()
    else:
        inner_assessor = pipe.assessor
    assessor, repairer = R.CachedAssessor(inner_assessor, cache), R.CachedRepairer(pipe.repairer, cache)
    cfg = lambda: PC.ControllerPolicyConfig(enable_repair=True)  # noqa: E731  (repair stays enabled end to end)
    oracle = lambda prep: R.OracleAssessor(prep["gold"])  # noqa: E731
    makers: Dict[str, Dict[int, Callable[[int, int], Any]]] = {}
    for s in seeds:
        tr_prep, _ = R.prepare(tr, R.stratified_sample(tr.responses, tr.sources, args.train_responses, s + 1000))
        ck = ROOT / "experiments/runs" / f"ragtruth_ablation_full_seed{s}_{args.ablation_ts}.json"
        lin = ContextualMABPolicy.load_checkpoint(ck, frozen=True)
        fr = PC.fit_feature_ranker(tr_prep, ["cts", "type"])
        ts_pol = PC.TypeThompson(seed=s, config=PC.ControllerPolicyConfig(enable_repair=False))
        for p in tr_prep:
            R.run_controller(p, ts_pol, 4, R.OracleAssessor(p["gold"]), R._CostedMockRepairer(), f"ts_{s}")
        profiles = {}
        for b in budgets:
            lin.config.enable_repair = False
            _, rows = R.run_setting("LinUCB", b, s, tr_prep, {"LinUCB": lin}, None, R._CostedMockRepairer(), None,
                                    assessor_factory=oracle, enable_repair=False, n_boot=0)
            profiles[b] = PC.estimate_position_profile(rows)
        lin.config.enable_repair = True
        makers.setdefault("LinUCB", {})[s] = (lambda l: lambda idx, b: copy.deepcopy(l))(lin)
        makers.setdefault("FeatureRanker(CTS+type)", {})[s] = (lambda m: lambda idx, b: PC.FeatureRanker(m, "FR", cfg()))(fr)
        makers.setdefault("TypeThompson", {})[s] = (lambda t, ss: lambda idx, b: t.frozen_copy(ss * 1_000_003 + idx))(ts_pol, s)
        makers.setdefault("PositionMatchedRandom", {})[s] = (lambda pr, ss: lambda idx, b: PC.PositionMatchedRandom(pr[b], ss * 1_000_003 + idx, cfg()))(profiles, s)
        makers.setdefault("Random", {})[s] = (lambda ss: lambda idx, b: RandomPolicy(cfg(), seed=ss * 1_000_003 + idx))(s)
        print(f"seed {s}: fitted LinUCB(loaded), ranker, TypeThompson, position profiles ({time.monotonic() - t0:.0f}s)", flush=True)
    from src.controller.policy import SequentialDecisionPolicy, StaticAllocationPolicy, StaticCTSRankedPolicy
    makers["StaticFIFO"] = {0: lambda idx, b: StaticAllocationPolicy(cfg())}
    makers["StaticCTSRanked"] = {0: lambda idx, b: StaticCTSRankedPolicy(cfg())}
    makers["LexicalSupport"] = {0: lambda idx, b: PC.LexicalSupportPolicy(cfg())}

    runs: Dict[tuple, List[List[Dict[str, Any]]]] = {}
    for b in budgets:
        for kind, by_seed in makers.items():
            seed_rows = []
            for s, mk in by_seed.items():
                rows = [run_one(p, mk(i, b), b, pipe.retriever, assessor, repairer, f"{kind}_b{b}_s{s}_{p['sample_id']}") for i, p in enumerate(preps)]
                seed_rows.append(rows)
                save_cache(cache)
            runs[(b, kind)] = seed_rows
            print(f"[budget {b}] {kind}: seeds {list(by_seed)} done ({time.monotonic() - t0:.0f}s; cache hits {cache.hits}, misses {cache.misses})", flush=True)

    # ---- summaries
    bases = [question_baselines(p) for p in preps]
    ret = np.array([b["retrieval_str_em"] for b in bases]); dr = np.array([b["draft_str_em"] for b in bases])
    summary: Dict[str, Any] = {"n_questions": len(preps), "retrieval_str_em": float(ret.mean()), "retrieval_ci95": boot_ci(ret),
                               "draft_str_em": float(dr.mean()), "draft_ci95": boot_ci(dr),
                               "mean_claims_per_draft": float(np.mean([len(p["claims"]) for p in preps])), "budgets": {}}
    for b in budgets:
        per_kind, final_q = {}, {}
        for kind in makers:
            seed_rows = runs[(b, kind)]
            f_q = aggregate({(b, kind): seed_rows}, bases, qids)[(b, kind)]
            nr_q = aggregate({(b, kind): seed_rows}, bases, qids, "str_em_final_no_repair")[(b, kind)]
            final_q[kind] = f_q
            flat = [r for rows in seed_rows for r in rows]
            agg = lambda k: float(np.mean([r[k] for r in flat]))  # noqa: E731
            per_kind[kind] = {"str_em_final": float(f_q.mean()), "ci95": boot_ci(f_q), "repair_gain": float((f_q - nr_q).mean()),
                              "n_seeds": len(seed_rows), "claims_verified": agg("n_verified"), "claims_checked": agg("n_checked"),
                              "flagged": agg("n_flagged"), "repairs": agg("n_repair"), "reverified_pass": agg("n_reverified_pass"),
                              "model_calls": agg("model_calls"), "tokens": agg("total_tokens"), "latency_seconds": agg("latency_seconds"),
                              "ungrounded_checked": agg("ungrounded_checked"), "ungrounded_flagged": agg("ungrounded_flagged"),
                              "grounded_checked": agg("grounded_checked"), "grounded_flagged": agg("grounded_flagged"),
                              "ungrounded_in_final": agg("ungrounded_in_final"), "ungrounded_total": agg("ungrounded_total"),
                              "selection_loss": float((dr - f_q).mean())}
        paired = {}
        for a in ("LinUCB", "FeatureRanker(CTS+type)"):
            for c in makers:
                if c != a:
                    d = final_q[a] - final_q[c]
                    paired[f"{a} - {c}"] = {"diff": float(d.mean()), "ci95": boot_ci(d)}
        summary["budgets"][str(b)] = {"policies": per_kind, "paired": paired}
    out = {"manifest": {"run_id": f"alce_e2e_{ts}{('_' + args.tag) if args.tag else ''}", "command": " ".join(sys.argv),
                        "collection": args.collection, "questions": f"ASQA[{args.start}:{args.start + args.n_questions}]", "seeds": seeds,
                        "budgets": budgets, "components": {"retriever": "QdrantRetriever (real)", "generator/repairer": "Llama-3.2-1B Q4 via llama-server (real)",
                                                           "assessor": ("NLI cross-encoder " + inner_assessor.model_name + " (real, fixed thresholds 0.5/0.5)") if args.verifier == "nli" else "Llama-3.2-1B Q4 via llama-server (real)",
                                                           "decomposer/CTS": "rule-based", "final answer": "deterministic stitch of verified claims (LLM rewrite skipped)"},
                        "policies_fit_on": "RAGTruth TRAIN split (oracle control), frozen, transferred to ASQA", "llm_cache": {"hits": cache.hits, "misses": cache.misses}},
           "summary": summary, "per_question_runs": {f"{b}|{k}": rows for (b, k), rows in runs.items()}}
    jp = out_dir / f"{out['manifest']['run_id']}.json"
    jp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {jp}")
    return out


if __name__ == "__main__":
    main()
