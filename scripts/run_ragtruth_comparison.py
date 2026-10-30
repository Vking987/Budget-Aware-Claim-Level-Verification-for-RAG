#!/usr/bin/env python3
"""RAGTruth multi-policy comparison under identical examples, evidence, and budgets.

Pipeline per response: RAGTruth response -> rule-based decomposition + CTS (existing adapter)
-> per-claim evidence = top-k lexically ranked chunks of the GOLD source document (oracle
context; no retrieval error) -> VerificationController with the policy under test, real
LlamaAssessor + real LLMRepairer, repair followed by re-verification (controller).

Fairness rules enforced here:
  * one seeded, task-type-stratified sample of the held-out TEST split, used by every policy;
  * identical claims, evidence chunks and budget grid for every policy (asserted);
  * a fresh policy instance per response (static policies keep per-episode state);
  * LinUCB is FROZEN at evaluation and trained only on the TRAIN split (--train-responses);
  * assessor/repairer calls are memoized on (claim text, evidence text[, verdict]) because both run
    at temperature 0: a cache hit replays the stored verdict AND its originally recorded cost, so
    budget accounting is identical to a real call and policies differ only in WHAT they verify.
    Cache hit/miss counts are written to the manifest.

Scoring (src/evaluation/ragtruth_metrics.py): verifier flags vs gold claim labels; CTS-triage
metrics are reported separately and do not depend on the policy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
import re
import socket
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controller.baselines import RandomPolicy, VerifyAllPolicy
from src.controller.controller import VerificationController
from src.controller.interfaces import BaseRepairer, LLMRepairer, MockRepairer, MockRetriever
from src.controller.mab_policy import ContextualMABPolicy, MABPolicyConfig
from src.controller.models import PerClaimBudget, VerificationBudget
from src.controller.policy import (
    ControllerPolicyConfig, SequentialDecisionPolicy, StaticAllocationPolicy, StaticCTSRankedPolicy,
)
from src.controller.reward import ProxyVerificationRewardModel
from src.evaluation import ragtruth_metrics as M
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter, compute_claim_evaluation_metrics
from src.evidence.assessor import BaseAssessor, MockDeterministicAssessor
from src.evidence.llama_assessor import LlamaAssessor
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost

DATA_DIR = PROJECT_ROOT / "data" / "ragtruth"
RUN_DIR = PROJECT_ROOT / "experiments" / "runs"
DEFAULT_CACHE = PROJECT_ROOT / "experiments" / "cache" / "ragtruth_llm_cache.pkl"
ALL_POLICIES = ["LinUCB", "StaticCTSRanked", "StaticFIFO", "DeterministicCTS", "Random", "VerifyAll"]
EVIDENCE_TOP_K = 4
CHUNK_CHARS = 700


class _CostedMockAssessor(MockDeterministicAssessor):
    """Dry-run only: charges 1 model call like the real assessor so budgets bind."""

    def assess(self, claim, passages):
        a = super().assess(claim, passages)
        a.cost = VerificationCost(model_calls=1, latency_seconds=0.0)
        return a


class OracleAssessor(BaseAssessor):
    """BENCHMARK CONTROL, not a verifier: returns the GOLD label for every claim (HALLUCINATED -> CONTRADICTED, anything
    else -> SUPPORTED) at the cost of one model call, so budgets bind exactly as with the real assessor. It removes
    verifier error so that policy SELECTION can be measured on its own. Ambiguous gold labels (UNCERTAIN,
    IMPLICIT_TRUE_ABSENT) come back SUPPORTED and are not scored. Always used with repair disabled."""

    def __init__(self, gold: Dict[str, str]) -> None:
        self.gold = gold

    @property
    def assessor_id(self) -> str:
        return "oracle_gold_labels"

    def assess(self, claim, passages):
        bad = self.gold.get(claim.claim_id) == "HALLUCINATED"
        return EvidenceAssessment(
            claim_id=claim.claim_id, evidence_ids=[p.evidence_id for p in passages], supporting_spans={},
            label=AssessmentLabel.CONTRADICTED if bad else AssessmentLabel.SUPPORTED, confidence=0.8,
            rationale="oracle (gold label)", cost=VerificationCost(model_calls=1, latency_seconds=0.0, model_id="oracle"),
        )


class _CostedMockRepairer(MockRepairer):
    def repair(self, claim, assessment, passages):
        r, _ = super().repair(claim, assessment, passages)
        return r, VerificationCost(model_calls=1, latency_seconds=0.0)


# ----------------------------------------------------------------------------- caching wrappers
class LLMCache:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self.data: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        if path and path.exists():
            with open(path, "rb") as f:
                self.data = pickle.load(f)

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f)
        tmp.replace(self.path)


def _key(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def _evidence_key(passages: List[EvidencePassage]) -> str:
    return "\x1e".join(f"{p.evidence_id}\x1d{p.source_text}" for p in passages)


class CachedAssessor(BaseAssessor):
    def __init__(self, inner: BaseAssessor, cache: LLMCache) -> None:
        self.inner, self.cache = inner, cache

    @property
    def assessor_id(self) -> str:
        return self.inner.assessor_id

    def assess(self, claim, passages):
        k = _key("assess", claim.normalized_text, _evidence_key(passages))
        hit = self.cache.data.get(k)
        if hit is not None:
            self.cache.hits += 1
            return replace(copy.deepcopy(hit), claim_id=claim.claim_id)
        self.cache.misses += 1
        res = self.inner.assess(claim, passages)
        if not getattr(res, "is_error", False):  # never memoize technical failures
            self.cache.data[k] = copy.deepcopy(res)
        return res


class CachedRepairer(BaseRepairer):
    def __init__(self, inner: BaseRepairer, cache: LLMCache) -> None:
        self.inner, self.cache = inner, cache

    def repair(self, claim, assessment, passages):
        label = assessment.label.value if assessment else ""
        k = _key("repair", claim.normalized_text, label, _evidence_key(passages))
        hit = self.cache.data.get(k)
        if hit is not None:
            self.cache.hits += 1
            res, cost = copy.deepcopy(hit)
            return replace(res, claim_id=claim.claim_id), cost
        self.cache.misses += 1
        res, cost = self.inner.repair(claim, assessment, passages)
        if res.proposed_revision:  # empty revision == parse/server failure; allow retry
            self.cache.data[k] = copy.deepcopy((res, cost))
        return res, cost


# ----------------------------------------------------------------------------- data preparation
_TOK = re.compile(r"[a-z0-9]+")


def chunk_source(text: str, max_chars: int = CHUNK_CHARS) -> List[str]:
    """Split source into <=max_chars chunks on line/sentence boundaries (policy independent)."""
    pieces: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        pieces.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    chunks: List[str] = []
    cur = ""
    for p in pieces:
        while len(p) > max_chars:  # hard-split very long pieces (e.g. JSON blobs)
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:max_chars])
            p = p[max_chars:]
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def claim_evidence(source_id: str, title: str, chunks: List[str], claim_text: str, k: int = EVIDENCE_TOP_K):
    ctoks = set(_TOK.findall(claim_text.lower()))
    scored = []
    for i, ch in enumerate(chunks):
        overlap = len(ctoks & set(_TOK.findall(ch.lower()))) / max(1, len(ctoks))
        scored.append((-overlap, i, ch))
    scored.sort()
    return [
        EvidencePassage(evidence_id=f"rt_{source_id}_c{i}", doc_title=title, source_text=ch,
                        retrieval_score=round(-neg, 4), metadata={"source_id": source_id})
        for neg, i, ch in scored[:k]
    ]


def stratified_sample(responses, sources, n: int, seed: int):
    by_task: Dict[str, list] = {}
    for r in responses:
        s = sources.get(r.source_id)
        if s is not None:
            by_task.setdefault(s.task_type, []).append(r)
    tasks = sorted(by_task)
    rng = random.Random(seed)
    base, rem = divmod(n, len(tasks))
    out = []
    for i, t in enumerate(tasks):
        pool = sorted(by_task[t], key=lambda r: r.response_id)
        out.extend(rng.sample(pool, min(len(pool), base + (1 if i < rem else 0))))
    return sorted(out, key=lambda r: r.response_id)


def prepare(adapter: RAGTruthDatasetAdapter, responses) -> Tuple[List[Dict[str, Any]], List[Any]]:
    prepared, all_mapped = [], []
    for r in responses:
        src = adapter.sources[r.source_id]
        mapped = adapter.mapper.map_response(r, src)
        claims = [m.claim_record for m in mapped if m.claim_record]
        if not claims:
            continue
        ids = [c.claim_id for c in claims]
        assert len(ids) == len(set(ids)), f"duplicate claim ids in {r.response_id}"
        chunks = chunk_source(src.get_source_text())
        title = src.source or f"Source_{src.source_id}"
        by_id = {m.claim_record.claim_id: m for m in mapped if m.claim_record}
        prepared.append({
            "resp": r, "task": src.task_type, "claims": claims,
            "gold": {cid: by_id[cid].mapped_label.value.upper() for cid in ids},
            "evidence": {c.claim_id: claim_evidence(src.source_id, title, chunks, c.normalized_text) for c in claims},
            "gold_response": bool(r.labels),
        })
        all_mapped.extend(mapped)
    return prepared, all_mapped


# ----------------------------------------------------------------------------- running policies
def make_policy(name: str, seed: int, idx: int, linucb: Dict[str, ContextualMABPolicy], enable_repair: bool = True):
    cfg = ControllerPolicyConfig(enable_repair=enable_repair)
    if name.startswith("LinUCB"):
        pol = copy.deepcopy(linucb[name])
        pol.config.enable_repair = enable_repair
        return pol
    if name == "StaticCTSRanked":
        return StaticCTSRankedPolicy(cfg)
    if name == "StaticFIFO":
        return StaticAllocationPolicy(cfg)
    if name == "DeterministicCTS":
        return SequentialDecisionPolicy(cfg)
    if name == "VerifyAll":
        return VerifyAllPolicy(cfg)
    if name == "Random":
        return RandomPolicy(cfg, seed=seed * 1_000_003 + idx)
    raise ValueError(name)


def run_controller(prep, policy, budget_calls, assessor, repairer, run_id):
    claims = copy.deepcopy(prep["claims"])
    retriever = MockRetriever()  # oracle context: returns the claim's pre-ranked gold-source chunks
    for c in claims:
        retriever.set_evidence_for_claim(c.claim_id, prep["evidence"][c.claim_id])
    ctl = VerificationController(
        retriever=retriever, assessor=assessor, repairer=repairer, policy=policy,
        global_budget=VerificationBudget(max_steps=50, max_model_calls=budget_calls),
        per_claim_budget=PerClaimBudget(),
    )
    trace = ctl.run(claims=claims, initial_evidence={c.claim_id: list(prep["evidence"][c.claim_id]) for c in claims},
                    run_id=run_id)
    return trace, ctl


def extract(prep, trace):
    first: Dict[str, Optional[str]] = {c.claim_id: None for c in prep["claims"]}
    errors = 0
    repairs = []
    for s in trace.steps:
        a = s.action.value
        if a == "VERIFY":
            if s.error or not s.assessment_label:
                errors += 1
            elif first.get(s.claim_id) is None:
                first[s.claim_id] = s.assessment_label
        elif a == "REPAIR" and s.repair_data is not None:
            rd = s.repair_data
            rv = rd.get("reverification")
            repairs.append({"gold": prep["gold"].get(s.claim_id), "revision_nonempty": bool((rd.get("proposed_revision") or "").strip()),
                            "reverified": (bool(rv["verified"]) if rv else None)})
    rid = prep["resp"].response_id
    rows = [{"response_id": rid, "claim_id": cid, "gold": prep["gold"][cid], "label": first[cid]} for cid in first]
    return rows, repairs, errors


def run_setting(policy_name, budget, seed, prepared, linucb, assessor, repairer, cache,
                assessor_factory=None, enable_repair=True, n_boot=1000):
    t0 = time.monotonic()
    rows, repairs, resp_rows = [], [], []
    cost = {"model_calls": 0, "total_tokens": 0, "latency_seconds": 0.0}
    steps = errors = 0
    action_counts: Dict[str, int] = {}
    for idx, prep in enumerate(prepared):
        pol = make_policy(policy_name, seed, idx, linucb, enable_repair)
        asr = assessor_factory(prep) if assessor_factory else assessor
        trace, _ = run_controller(prep, pol, budget, asr, repairer, f"{policy_name}_b{budget}_{prep['resp'].response_id}")
        r, rep, err = extract(prep, trace)
        rows += r
        repairs += rep
        errors += err
        steps += trace.total_steps
        tc = trace.total_cost if isinstance(trace.total_cost, dict) else trace.total_cost.to_dict()
        cost["model_calls"] += tc.get("model_calls") or 0
        cost["total_tokens"] += tc.get("total_tokens") or 0
        cost["latency_seconds"] += tc.get("latency_seconds") or 0.0
        for s in trace.steps:
            action_counts[s.action.value] = action_counts.get(s.action.value, 0) + 1
        resp_rows.append({"gold": prep["gold_response"], "labels": [x["label"] for x in r]})
    counts = M.claim_counts_by_response(rows)
    out: Dict[str, Any] = {"policy": policy_name, "budget_model_calls": budget, "seed": seed,
                           "n_responses": len(prepared), "n_claims": len(rows),
                           "n_excluded_gold": sum(1 for x in rows if x["gold"] not in M.SCORED_GOLD),
                           "n_claims_verified": sum(1 for x in rows if x["label"] is not None),
                           "assessor_errors": errors, "steps": steps, "action_counts": action_counts,
                           "recorded_cost": cost, "wall_clock_seconds": round(time.monotonic() - t0, 2)}
    for mode in ("strict", "broad"):
        p = M.pooled(counts.values(), mode)
        out[f"claim_{mode}"] = {**p, **{f"{m}_ci95": M.bootstrap_ci(counts, mode, m, n_boot=n_boot, seed=seed) for m in ("precision", "recall", "f1")}}
        out[f"response_{mode}"] = M.response_level(resp_rows, mode)
        out[f"hallucinations_detected_per_model_call_{mode}"] = (p["tp"] / cost["model_calls"]) if cost["model_calls"] else None
    out["repair"] = M.repair_summary(repairs)
    # SELECTION quality, independent of what the verifier concluded: was each gold positive verified at all?
    sel = M.pooled(counts.values(), "sel")
    n_pos, n_scored = sel["tp"] + sel["fn"], sum(sel[k] for k in ("tp", "fp", "fn", "tn"))
    base_rate = n_pos / n_scored if n_scored else None
    out["selection"] = {
        "gold_positives": n_pos, "verified_positives": sel["tp"], "verified_negatives": sel["fp"],
        "selection_recall": sel["recall"],
        "selection_recall_ci95": M.bootstrap_ci(counts, "sel", "recall", n_boot=n_boot, seed=seed),
        "coverage": sel["flagged_fraction"],
        "coverage_ci95": M.bootstrap_ci(counts, "sel", "flagged_fraction", n_boot=n_boot, seed=seed),
        "lift_over_base_rate": (sel["precision"] / base_rate) if (sel["precision"] is not None and base_rate) else None,
    }
    return out, rows


# ----------------------------------------------------------------------------- LinUCB training (train split only)
def train_linucb(prepared, assessor, repairer, budget, seed, out_path: Path, assessor_factory=None, enable_repair=True):
    policy = ContextualMABPolicy(config=MABPolicyConfig(alpha=0.5, frozen=False, random_seed=seed, enable_repair=enable_repair),
                                 reward_model=ProxyVerificationRewardModel())
    for prep in prepared:
        asr = assessor_factory(prep) if assessor_factory else assessor
        run_controller(prep, policy, budget, asr, repairer, f"train_{prep['resp'].response_id}")
        policy.mark_episode_end(f"train_{prep['resp'].response_id}")
    policy.save_checkpoint(out_path, training_split="ragtruth_train", random_seed=seed,
                           metadata={"n_train_responses": len(prepared), "train_budget_model_calls": budget,
                                     "reward": "ProxyVerificationRewardModel (computed from assessor output, not from gold labels)",
                                     "verifier": "oracle (selection-only control)" if assessor_factory else "LlamaAssessor",
                                     "enable_repair": enable_repair})
    return policy


# ----------------------------------------------------------------------------- reporting
def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def markdown_table(results) -> str:
    lines = []
    for mode in ("broad", "strict"):
        lines += [f"\n### Verifier claim-level ({mode} flag), mean over seeds where stochastic\n",
                  "| Policy | Budget (calls) | Precision | Recall | F1 | F1 95% CI | TP/model call | Calls used | Repair re-verified / triggered |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in results:
            c = r[f"claim_{mode}"]
            ci = c["f1_ci95"]
            rp = r["repair"]
            lines.append(f"| {r['policy']} (s{r['seed']}) | {r['budget_model_calls']} | {fmt(c['precision'])} | {fmt(c['recall'])} | "
                         f"{fmt(c['f1'])} | {('[%.3f, %.3f]' % tuple(ci)) if ci else 'n/a'} | "
                         f"{fmt(r[f'hallucinations_detected_per_model_call_{mode}'])} | {r['recorded_cost']['model_calls']} | "
                         f"{rp['reverified_pass']}/{rp['repairs_triggered']} |")
    return "\n".join(lines)


def preflight_llm() -> None:
    import os
    u = urlparse(os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1"))
    try:
        socket.create_connection((u.hostname or "127.0.0.1", u.port or 8080), timeout=3).close()
    except OSError:
        sys.exit(f"ERROR: llama-server not reachable at {u.hostname}:{u.port}. Start it (or use --dry-run for plumbing only).")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--allow-non-test", action="store_true", help="permit evaluating on a non-test split (debug only)")
    ap.add_argument("--n", type=int, default=10, help="responses to evaluate (stratified by task type)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", default="2,4,8", help="comma-separated global max_model_calls per response")
    ap.add_argument("--policies", default=",".join(ALL_POLICIES))
    ap.add_argument("--random-seeds", default="0,1,2")
    ap.add_argument("--linucb", action="append", default=[], metavar="NAME=CKPT",
                    help="frozen LinUCB checkpoint(s); NAME must start with 'LinUCB'. Default: trained one from --train-responses")
    ap.add_argument("--train-responses", type=int, default=0, help="train LinUCB on this many TRAIN-split responses first")
    ap.add_argument("--train-budget", type=int, default=4)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--fixture", default=None, help="tiny fixture JSON (tests/dry-run plumbing only)")
    ap.add_argument("--dry-run", action="store_true", help="MOCK assessor/repairer: plumbing only, results meaningless")
    ap.add_argument("--out-dir", default=str(RUN_DIR))
    ap.add_argument("--tag", default="")
    ap.add_argument("--verifier", choices=["llm", "oracle"], default="llm",
                    help="llm (default): real LlamaAssessor + LLMRepairer. oracle: gold-label control that removes verifier error to "
                         "measure policy SELECTION only; repair is disabled; NOT a verifier result")
    ap.add_argument("--all-responses", action="store_true", help="evaluate every response of the split instead of a seeded sample")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for confidence intervals")
    args = ap.parse_args(argv)

    if args.split != "test" and not args.allow_non_test:
        sys.exit("Refusing to evaluate on a non-test split without --allow-non-test.")
    oracle = args.verifier == "oracle"
    if not args.dry_run and not oracle:
        preflight_llm()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assessor_factory, enable_repair = None, True
    if oracle:  # gold-label control: no LLM, no cache, repair disabled for every policy
        cache = LLMCache(None)
        inner_a, inner_r = OracleAssessor({}), _CostedMockRepairer()
        assessor, repairer = inner_a, inner_r
        assessor_factory, enable_repair = (lambda prep: OracleAssessor(prep["gold"])), False
    else:
        cache = LLMCache(None if (args.no_cache or args.dry_run) else Path(args.cache))
        inner_a = _CostedMockAssessor() if args.dry_run else LlamaAssessor()
        inner_r = _CostedMockRepairer() if args.dry_run else LLMRepairer()
        assessor, repairer = CachedAssessor(inner_a, cache), CachedRepairer(inner_r, cache)

    # --- data
    def load(split):
        ad = RAGTruthDatasetAdapter()
        if args.fixture:
            ad.load_from_fixture(args.fixture)
        else:
            ad.load_from_jsonl(DATA_DIR / "source_info.jsonl", DATA_DIR / "response.jsonl", split=split)
        return ad

    linucb: Dict[str, ContextualMABPolicy] = {}
    train_info = None
    if args.train_responses > 0:
        tr = load("train")
        tr_sample = stratified_sample(tr.responses, tr.sources, args.train_responses, args.seed + 1000)
        tr_prep, _ = prepare(tr, tr_sample)
        ckpt = out_dir / f"ragtruth_linucb_train{len(tr_prep)}_{ts}.json"
        t0 = time.monotonic()
        linucb["LinUCB"] = train_linucb(tr_prep, assessor, repairer, args.train_budget, args.seed, ckpt,
                                        assessor_factory=assessor_factory, enable_repair=enable_repair)
        linucb["LinUCB"].freeze() if hasattr(linucb["LinUCB"], "freeze") else None
        cache.save()
        train_info = {"checkpoint": str(ckpt), "n_responses": len(tr_prep), "split": "train", "budget": args.train_budget,
                      "seconds": round(time.monotonic() - t0, 1),
                      "response_ids": [p["resp"].response_id for p in tr_prep]}
    for spec in args.linucb:
        name, path = spec.split("=", 1)
        assert name.startswith("LinUCB"), "--linucb NAME must start with 'LinUCB'"
        linucb[name] = ContextualMABPolicy.load_checkpoint(path, frozen=True)

    ad = load(args.split)
    if args.all_responses:
        sample = sorted((r for r in ad.responses if r.source_id in ad.sources), key=lambda r: r.response_id)
    else:
        sample = stratified_sample(ad.responses, ad.sources, args.n, args.seed)
    prepared, mapped = prepare(ad, sample)
    if train_info:
        overlap = set(train_info["response_ids"]) & {p["resp"].response_id for p in prepared}
        assert not overlap, f"train/eval overlap: {sorted(overlap)[:5]}"

    cts = compute_claim_evaluation_metrics(mapped, [p["resp"] for p in prepared]).to_dict()

    # --- policies x budgets
    pol_names = [p for p in args.policies.split(",") if p]
    expanded: List[Tuple[str, int]] = []
    for p in pol_names:
        if p == "LinUCB":
            expanded += [(n, args.seed) for n in linucb] if linucb else []
        elif p == "Random":
            expanded += [("Random", int(s)) for s in args.random_seeds.split(",")]
        else:
            expanded.append((p, args.seed))
    if "LinUCB" in pol_names and not linucb:
        sys.exit("LinUCB requested but no checkpoint: pass --train-responses K or --linucb LinUCB=path")
    budgets = [int(b) for b in args.budgets.split(",")]

    results, all_rows, ref = [], {}, None
    for b in budgets:
        for name, seed in expanded:
            res, rows = run_setting(name, b, seed, prepared, linucb, assessor, repairer, cache,
                                    assessor_factory=assessor_factory, enable_repair=enable_repair, n_boot=args.n_boot)
            ident = [(r["response_id"], r["claim_id"], r["gold"]) for r in rows]
            if ref is None:
                ref = ident
            assert ident == ref, f"{name} (budget {b}) did not see the identical claims as the reference run"
            results.append(res)
            all_rows[f"{name}|b{b}|s{seed}"] = rows
            cache.save()
            print(f"[{name:<22} budget={b} seed={seed}] claims={res['n_claims']} verified={res['n_claims_verified']} "
                  f"calls={res['recorded_cost']['model_calls']} broad F1={fmt(res['claim_broad']['f1'])} "
                  f"strict F1={fmt(res['claim_strict']['f1'])} sel.recall={fmt(res['selection']['selection_recall'])} "
                  f"wall={res['wall_clock_seconds']}s", flush=True)

    manifest = {
        "run_id": f"ragtruth_comparison_{ts}{('_' + args.tag) if args.tag else ''}",
        "git_head": git_head(), "command": " ".join(sys.argv), "dry_run": args.dry_run,
        "protocol": {
            "split": args.split, "sampling": "all responses of the split" if args.all_responses else "stratified by task type, seeded",
            "verifier": args.verifier, "repair_enabled": enable_repair, "n_boot": args.n_boot,
            "seed": args.seed, "n_requested": args.n,
            "n_responses": len(prepared), "response_ids": [p["resp"].response_id for p in prepared],
            "task_counts": {t: sum(1 for p in prepared if p["task"] == t) for t in sorted({p["task"] for p in prepared})},
            "gold_response_hallucinated": sum(1 for p in prepared if p["gold_response"]),
            "budget_unit": "global max_model_calls per response (max_steps=50 non-binding; per-claim 2 retrievals/2 verifications/1 repair)",
            "evidence": f"top-{EVIDENCE_TOP_K} lexically ranked {CHUNK_CHARS}-char chunks of the gold source per claim (oracle context)",
            "flag_definitions": {"strict": sorted(M.STRICT), "broad": "any label except SUPPORTED"},
            "unverified_claims": "counted as not flagged", "linucb_training": train_info,
            "linucb_checkpoints_evaluated": {n: ("trained-here" if n == "LinUCB" else "external") for n in linucb},
        },
        "components": {
            "assessor": {"class": type(inner_a).__name__, "real": not (args.dry_run or oracle),
                         **({"note": "gold-label control; measures policy selection with verifier error removed; not a verifier"} if oracle else {})},
            "repairer": {"class": type(inner_r).__name__, "real": not (args.dry_run or oracle),
                         **({"note": "never invoked: repair disabled in oracle mode"} if oracle else {})},
            "retriever": {"class": "MockRetriever", "real": False, "note": "serves pre-ranked gold-source chunks (oracle context)"},
            "decomposer_cts": {"class": "RuleBasedDecomposer/ClaimTriageScorer", "real": True, "note": "rule-based"},
            "llm_cache": {"enabled": cache.path is not None, "hits": cache.hits, "misses": cache.misses},
        },
        "cts_triage_metrics_policy_independent": cts,
    }
    out = {"manifest": manifest, "results": results}
    jp = out_dir / f"{manifest['run_id']}.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    if sum(len(v) for v in all_rows.values()) > 150_000:  # keep very large row files compact
        import gzip
        with gzip.open(out_dir / f"{manifest['run_id']}_claim_rows.json.gz", "wt", encoding="utf-8") as f:
            json.dump(all_rows, f)
    else:
        with open(out_dir / f"{manifest['run_id']}_claim_rows.json", "w", encoding="utf-8") as f:
            json.dump(all_rows, f)
    md = out_dir / f"{manifest['run_id']}.md"
    md.write_text(f"# {manifest['run_id']}\n\nDry-run (mock): {args.dry_run}\n" + markdown_table(results), encoding="utf-8")
    print(f"\nWrote {jp}\n      {md}")
    return out


if __name__ == "__main__":
    main()
