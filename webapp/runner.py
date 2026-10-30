"""Streaming runner for Adaptive RAG Lab.

Runs the repository's REAL components (retriever, draft generator, content filter, decomposer, CTS scorer, VerificationController with the chosen
policy, assessor, repairer, grounding/stitching) and emits one event per real backend step. Nothing is simulated: the policy, assessor, repairer and
retriever are wrapped by thin observers that report each call, and the controller itself is unchanged. The verification budget is enforced by the
controller's BudgetTracker (max_model_calls); `budget_check` events report what it actually consumed.

Event shapes (all JSON): {"type": "stage", "stage": id, "status": "running|done|error|skipped", "duration_s": float|None, "data": {...}},
{"type": "round", ...} (one per policy decision), {"type": "verify", ...}, {"type": "repair", ...}, {"type": "feedback", ...}, {"type": "budget", ...},
{"type": "result", ...}, {"type": "error", ...}.
"""

from __future__ import annotations

import copy
import dataclasses
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus
from src.controller.baselines import RandomPolicy
from src.controller.budget import PerClaimBudget, VerificationBudget
from src.controller.controller import VerificationController
from src.controller.mab_policy import ContextualMABPolicy
from src.controller.models import ControllerAction
from src.controller.policy import ControllerPolicyConfig, StaticAllocationPolicy, StaticCTSRankedPolicy
from src.evidence.models import AssessmentLabel, EvidencePassage, VerificationCost
from src.synthesis.grounding import build_items, stitch_answer, support_for_claim
from webapp.generation import extractive_draft, is_insufficient, is_well_formed
from webapp.relevance import analyse_question, claim_is_on_topic, passages_support_question

ROOT = Path(__file__).resolve().parent.parent
LINUCB_CHECKPOINT = ROOT / "experiments/runs/ragtruth_ablation_full_seed0_20261029_002617.json"
STAGES = [
    ("question", "Question received"), ("retrieval", "Relevant documents retrieved"), ("draft", "Draft answer generated"),
    ("decomposition", "Draft decomposed into atomic claims"), ("scoring", "Claim features and CTS priorities computed"),
    ("selection", "Policy selects the next claim"), ("verification", "Selected claim verified against retrieved evidence"),
    ("feedback", "Feedback / reward processed, policy state"), ("repair", "Repair and re-verification"), ("answer", "Final answer reconstructed"),
]
ANSWER_RULES = {
    "S0": "S0 current: keep SUPPORTED claims and re-verified repairs; drop INSUFFICIENT and CONTRADICTED",
    "S1": "S1 conservative: keep SUPPORTED and INSUFFICIENT claims (marked); drop CONTRADICTED; no repairs",
    "S2": "S2 repair-first: S1 plus repairs that passed re-verification",
}
POLICY_INFO = {
    "LinUCB": {"label": "LinUCB (contextual bandit, frozen)", "learns": False,
               "note": "Disjoint LinUCB loaded from a frozen checkpoint trained offline (RAGTruth, oracle verdicts). Scores are theta.x + alpha*sqrt(x'A^-1 x) per (claim, action) arm. "
                       "Parameters are NOT updated during this run; the reward shown is computed but unused."},
    "StaticCTSRanked": {"label": "Static CTS-ranked (heuristic)", "learns": False,
                        "note": "Contextual scoring without learning: claims are ranked ONCE by Claim Triage Score and verified top-down."},
    "StaticFIFO": {"label": "Static FIFO (arrival order)", "learns": False, "note": "Verifies claims in the order they appear in the draft. No scores, no learning."},
    "Random": {"label": "Random (seeded)", "learns": False, "note": "Uniform choice among eligible (claim, action) pairs; seeded. No scores, no learning."},
}


class RunCancelled(Exception):
    pass


class InsufficientEvidence(Exception):
    """The retrieved passages cannot answer the question; the run stops with an explicit message instead of answering from unrelated text."""

    def __init__(self, message: str, data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.data = data or {}


@dataclass
class RunConfig:
    question: str
    source: str = "wikipedia"
    budget: int = 4
    policy: str = "LinUCB"
    verifier: str = "nli"
    answer_rule: str = "S0"
    seed: int = 0
    top_k: int = 5
    reuse_prep: bool = True


def passage_url(p: EvidencePassage) -> Tuple[Optional[str], str]:
    """(url, provenance) - a stored URL is used as is; otherwise a Wikipedia link is DERIVED from the title (flagged as derived)."""
    url = (p.metadata or {}).get("url")
    if url:
        return url, "source"
    if p.doc_title and p.doc_title != "Untitled":
        return "https://en.wikipedia.org/wiki/" + quote(p.doc_title.replace(" ", "_")), "derived from title"
    return None, "none"


def passage_dict(p: EvidencePassage, used_by_verifier: Optional[bool] = None) -> Dict[str, Any]:
    url, prov = passage_url(p)
    d = {"evidence_id": p.evidence_id, "title": p.doc_title, "text": p.source_text, "score": p.retrieval_score, "url": url, "url_provenance": prov,
         "score_kind": (p.metadata or {}).get("score_kind") or ("embedding cosine similarity" if p.retrieval_score is not None else None)}
    if used_by_verifier is not None:
        d["seen_by_verifier"] = used_by_verifier
    return d


def r(x, nd=4):
    return None if x is None else round(float(x), nd)


# ----------------------------------------------------------------------------- observers
class Memo:
    """In-process memo of deterministic verifier/repairer outputs (temperature 0 / fixed model), so repeated runs on the same draft are fast.
    A memo hit is flagged `cached` in events and reports the ORIGINAL recorded latency."""

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.lock = threading.Lock()


def _ev_key(passages) -> str:
    return "\x1e".join(f"{p.evidence_id}\x1d{p.source_text}" for p in passages)


class ObservedAssessor:
    def __init__(self, inner, memo: Memo, emit, ctx: Dict[str, Any], cancel: threading.Event):
        self.inner, self.memo, self.emit, self.ctx, self.cancel = inner, memo, emit, ctx, cancel

    @property
    def assessor_id(self):
        return self.inner.assessor_id

    def assess(self, claim, passages):
        if self.cancel.is_set():
            raise RunCancelled()
        reverif = claim.claim_id in self.ctx["repaired"]
        self.emit({"type": "verify", "phase": "start", "claim_id": claim.claim_id, "claim_text": claim.normalized_text, "reverification": reverif,
                   "evidence": [p.evidence_id for p in passages]})
        key = f"{self.inner.assessor_id}\x1f{claim.normalized_text}\x1f{_ev_key(passages)}"
        t0 = time.monotonic()
        with self.memo.lock:
            hit = self.memo.data.get(key)
        cached = hit is not None
        a = dataclasses.replace(copy.deepcopy(hit), claim_id=claim.claim_id) if cached else self.inner.assess(claim, passages)
        if not cached and not getattr(a, "is_error", False):
            with self.memo.lock:
                self.memo.data[key] = copy.deepcopy(a)
        titles = {p.evidence_id: p for p in passages}
        spans = [{"evidence_id": eid, "title": titles[eid].doc_title if eid in titles else None, "text": t} for eid, ts in (a.supporting_spans or {}).items() for t in ts]
        c = a.cost
        self.emit({"type": "verify", "phase": "done", "claim_id": claim.claim_id, "claim_text": claim.normalized_text, "reverification": reverif,
                   "label": a.label.value, "confidence": r(a.confidence), "rationale": a.rationale, "supporting_spans": spans[:3], "is_error": bool(getattr(a, "is_error", False)),
                   "assessor": self.inner.assessor_id, "cached": cached, "wall_s": r(time.monotonic() - t0), "recorded_latency_s": r(c.latency_seconds if c else None),
                   "model_calls": c.model_calls if c else None, "tokens": c.total_tokens if c else None, "token_kind": "encoder tokens" if "nli" in self.inner.assessor_id else "LLM prompt+completion tokens"})
        return a


class ObservedRepairer:
    def __init__(self, inner, memo: Memo, emit, ctx: Dict[str, Any], cancel: threading.Event):
        self.inner, self.memo, self.emit, self.ctx, self.cancel = inner, memo, emit, ctx, cancel

    def repair(self, claim, assessment, passages):
        if self.cancel.is_set():
            raise RunCancelled()
        self.emit({"type": "repair", "phase": "start", "claim_id": claim.claim_id, "claim_text": claim.normalized_text, "verdict": assessment.label.value if assessment else None})
        key = f"{type(self.inner).__name__}\x1f{claim.normalized_text}\x1f{assessment.label.value if assessment else ''}\x1f{_ev_key(passages)}"
        with self.memo.lock:
            hit = self.memo.data.get(key)
        cached = hit is not None
        t0 = time.monotonic()
        if cached:
            res, cost = copy.deepcopy(hit)
            res = dataclasses.replace(res, claim_id=claim.claim_id)
        else:
            res, cost = self.inner.repair(claim, assessment, passages)
            if res.proposed_revision:
                with self.memo.lock:
                    self.memo.data[key] = copy.deepcopy((res, cost))
        self.ctx["repaired"].add(claim.claim_id)
        self.emit({"type": "repair", "phase": "done", "claim_id": claim.claim_id, "original": claim.normalized_text, "revision": res.proposed_revision,
                   "reason": res.repair_reason, "repairer": type(self.inner).__name__, "cached": cached, "wall_s": r(time.monotonic() - t0),
                   "recorded_latency_s": r(cost.latency_seconds if cost else None), "model_calls": cost.model_calls if cost else None})
        return res, cost


class ObservedRetriever:
    def __init__(self, inner, emit, cancel):
        self.inner, self.emit, self.cancel = inner, emit, cancel

    def retrieve(self, claim, top_k: int = 3):
        if self.cancel.is_set():
            raise RunCancelled()
        t0 = time.monotonic()
        ps, cost = self.inner.retrieve(claim, top_k=top_k)
        self.emit({"type": "retrieve", "claim_id": claim.claim_id, "n": len(ps), "wall_s": r(time.monotonic() - t0), "model_calls": cost.model_calls if cost else None,
                   "passages": [passage_dict(p) for p in ps]})
        return ps, cost


class ObservedPolicy:
    """Delegates to the real policy; reports every decision (with the policy's own scores) and every feedback call."""

    def __init__(self, inner, name: str, emit, ctx, cancel):
        self.inner, self.name, self.emit, self.ctx, self.cancel = inner, name, emit, ctx, cancel
        self.config = inner.config
        self.round = 0

    def select_next_action(self, claims, evidence_store, assessments, repairs, budget_tracker):
        if self.cancel.is_set():
            raise RunCancelled()
        self.round += 1
        action, claim, rationale = self.inner.select_next_action(claims=claims, evidence_store=evidence_store, assessments=assessments, repairs=repairs, budget_tracker=budget_tracker)
        rem, cost = budget_tracker.get_remaining_budget(), budget_tracker.get_global_cost()
        self.emit({"type": "budget", "round": self.round, "model_calls_used": cost.get("model_calls", 0), "remaining": rem, "steps_taken": budget_tracker.steps_taken})
        active = [c for c in claims if c.state not in (ClaimState.VERIFIED, ClaimState.UNRESOLVED, ClaimState.DEFERRED)]
        cands: List[Dict[str, Any]] = []
        if action == ControllerAction.STOP:
            # nothing was scored: the policy stopped (budget exhausted or nothing eligible); never display invented candidates
            extra = {"frozen": True, "tie_break": None, "score_kind": "no candidates evaluated: the policy stopped"}
        elif self.name == "LinUCB" and getattr(self.inner, "last_decision", None):
            ld = self.inner.last_decision
            for c in sorted(ld["candidates"], key=lambda c: (-round(c["ucb"], 6), -round(c["cts"], 4), c["claim_id"], c["action"])):
                cands.append({"claim_id": c["claim_id"], "action": c["action"], "features": {k: r(v) for k, v in c["features"].items()}, "mean": r(c["mean"]),
                              "bonus": r(c["bonus"]), "score": r(c["ucb"]), "score_kind": "UCB = mean + bonus", "cts": r(c["cts"])})
            extra = {"alpha": ld["alpha"], "frozen": ld["frozen"], "tie_break": ld["tie_break"]}
        elif self.name == "LinUCB":
            extra = {"alpha": None, "frozen": True, "tie_break": None, "score_kind": "LinUCB scored no candidate in this round"}
        elif self.name == "StaticCTSRanked":
            ranking = list(getattr(self.inner, "_fixed_ranking", None) or [])
            for c in active:
                cands.append({"claim_id": c.claim_id, "action": None, "score": r(c.triage_score), "score_kind": "CTS (fixed ranking key; rank " + (str(ranking.index(c.claim_id) + 1) if c.claim_id in ranking else "?") + ")", "cts": r(c.triage_score)})
            cands.sort(key=lambda c: (ranking.index(c["claim_id"]) if c["claim_id"] in ranking else 99))
            extra = {"frozen": True, "tie_break": "CTS desc, temporal claims first, claim_id"}
        elif self.name == "StaticFIFO":
            order = {c.claim_id: i for i, c in enumerate(claims)}
            for c in active:
                cands.append({"claim_id": c.claim_id, "action": None, "score": None, "score_kind": f"arrival index {order[c.claim_id] + 1}", "cts": r(c.triage_score)})
            extra = {"frozen": True, "tie_break": "arrival order"}
        else:
            for c in active:
                cands.append({"claim_id": c.claim_id, "action": None, "score": None, "score_kind": "no score (uniform over eligible pairs)", "cts": r(c.triage_score)})
            extra = {"frozen": True, "tie_break": "seeded uniform random"}
        self.emit({"type": "round", "round": self.round, "policy": self.name, "candidates": cands, "chosen": {"claim_id": claim.claim_id if claim else None, "action": action.value},
                   "rationale": rationale, "model_calls_used": cost.get("model_calls", 0), "model_calls_remaining": rem.get("remaining_model_calls"), **extra})
        return action, claim, rationale

    def record_feedback(self, *args, **kwargs):
        claim, action = kwargs.get("claim"), kwargs.get("action")
        reward = self.inner.record_feedback(*args, **kwargs)
        learns = self.name == "LinUCB"
        self.emit({"type": "feedback", "claim_id": claim.claim_id if claim else None, "action": action.value if action else None,
                   "reward": r(reward) if isinstance(reward, (int, float)) else None, "policy_learns": learns,
                   "policy_updated": bool(learns and not getattr(self.inner, "frozen", True)),
                   "note": ("reward computed by the proxy reward model from the verifier verdict; policy is frozen, so no parameter update" if learns and getattr(self.inner, "frozen", True)
                            else "this policy does not learn; no reward is computed" if not learns else "reward used to update the LinUCB arm")})
        return reward


# ----------------------------------------------------------------------------- answer reconstruction
def claim_outcomes(claims, trace) -> Dict[str, Any]:
    last, repairs_all, repairs_pass = {}, {}, {}
    for s in trace.steps:
        if s.action == ControllerAction.VERIFY and s.assessment_label and not s.error:
            last[s.claim_id] = s.assessment_label
        if s.action == ControllerAction.REPAIR and s.repair_data:
            repairs_all[s.claim_id] = s.repair_data
            if (s.repair_data.get("reverification") or {}).get("verified"):
                repairs_pass[s.claim_id] = s.repair_data
    return {"last": last, "verified": {c.claim_id for c in claims if c.state == ClaimState.VERIFIED}, "repairs_all": repairs_all, "repairs_pass": repairs_pass}


def claim_status(cid: str, out: Dict[str, Any], claims_by_id) -> Tuple[str, str]:
    """(status code, plain explanation). Status codes: supported, repaired, insufficient, contradicted, repair_failed, unverified."""
    last = out["last"].get(cid)
    if cid in out["repairs_pass"]:
        return "repaired", "Contradicted by the verifier, then repaired; the revision passed re-verification."
    if cid in out["repairs_all"]:
        return "repair_failed", "Contradicted by the verifier; a repair was attempted but did not pass re-verification."
    if last == AssessmentLabel.SUPPORTED.value:
        return "supported", "Verifier verdict SUPPORTED by the retrieved evidence (a verdict, not ground truth)."
    if last == AssessmentLabel.INSUFFICIENT.value:
        return "insufficient", "Verifier found the retrieved evidence insufficient to support or contradict the claim."
    if last in (AssessmentLabel.CONTRADICTED.value, AssessmentLabel.CONFLICTED.value):
        return "contradicted", "Verifier verdict CONTRADICTED/CONFLICTED by the retrieved evidence."
    st = claims_by_id[cid].state
    return "unverified", f"Not verified (state {st.value}): the call budget ended or the policy did not select it."


def assemble(rule: str, claims, out: Dict[str, Any], store) -> Tuple[str, List[Dict[str, Any]]]:
    verified, last, rp = out["verified"], out["last"], out["repairs_pass"]
    insufficient = {cid for cid, lab in last.items() if lab == AssessmentLabel.INSUFFICIENT.value and cid not in verified}
    if rule == "S0":
        ids, repairs = verified | set(rp), rp
    elif rule == "S1":
        ids, repairs = verified | insufficient, {}
    elif rule == "S2":
        ids, repairs = verified | insufficient | set(rp), rp
    else:
        raise ValueError(f"unknown answer rule {rule}")
    items = build_items(claims, repairs, store, ids)
    return stitch_answer(items), items


# ----------------------------------------------------------------------------- the run
class Backend:
    """Holds shared, lazily-created components. Constructor arguments exist so tests can inject scripted components (no network, no models)."""

    def __init__(self, retriever_factory=None, assessor_factory=None, repairer=None, draft_fn=None, policy_factory=None, pipeline_factory=None):
        self.retriever_factory, self.assessor_factory, self._repairer = retriever_factory, assessor_factory, repairer
        self.draft_fn, self.policy_factory, self.pipeline_factory = draft_fn, policy_factory, pipeline_factory
        self.memo = Memo()
        self.prep_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._retrievers: Dict[str, Any] = {}
        self._assessors: Dict[str, Any] = {}
        self._pipeline = None
        self._embed = None
        self.lock = threading.Lock()  # one pipeline run at a time (CPU bound)

    # -- component creation
    def retriever(self, source: str):
        if source not in self._retrievers:
            if self.retriever_factory:
                self._retrievers[source] = self.retriever_factory(source)
            elif source == "wikipedia":
                from webapp.wikipedia_retriever import WikipediaRetriever
                self._retrievers[source] = WikipediaRetriever()
            else:
                from qdrant_client import QdrantClient
                from src.controller.retriever import QdrantRetriever
                if self._embed is None:
                    from sentence_transformers import SentenceTransformer
                    self._embed = SentenceTransformer("BAAI/bge-large-en-v1.5")
                self._retrievers[source] = QdrantRetriever(client=QdrantClient(host="localhost", port=6333, timeout=60), embedding_model=self._embed, collection_name=SOURCES[source]["collection"])
        return self._retrievers[source]

    def assessor(self, name: str):
        if name not in self._assessors:
            if self.assessor_factory:
                self._assessors[name] = self.assessor_factory(name)
            elif name == "nli":
                from src.evidence.nli_assessor import NLIAssessor
                self._assessors[name] = NLIAssessor()
            else:
                from src.evidence.llama_assessor import LlamaAssessor
                self._assessors[name] = LlamaAssessor()
        return self._assessors[name]

    def repairer(self):
        if self._repairer is None:
            from src.controller.interfaces import LLMRepairer
            self._repairer = LLMRepairer()
        return self._repairer

    def pipeline(self):
        if self._pipeline is None:
            if self.pipeline_factory:
                self._pipeline = self.pipeline_factory()
            else:
                from src.pipeline import RAGVerificationPipeline
                # only the pipeline's own helpers are used (draft prompt, content filter, decomposer, CTS); all its components are lazy
                self._pipeline = RAGVerificationPipeline(long_answer=True)
        return self._pipeline

    def make_policy(self, cfg: RunConfig):
        if self.policy_factory:
            return self.policy_factory(cfg)
        c = ControllerPolicyConfig(enable_repair=True)
        if cfg.policy == "LinUCB":
            if not LINUCB_CHECKPOINT.exists():
                raise FileNotFoundError(f"LinUCB checkpoint not found: {LINUCB_CHECKPOINT}")
            p = ContextualMABPolicy.load_checkpoint(LINUCB_CHECKPOINT, frozen=True)
            p.config.enable_repair = True
            return p
        if cfg.policy == "StaticCTSRanked":
            return StaticCTSRankedPolicy(c)
        if cfg.policy == "StaticFIFO":
            return StaticAllocationPolicy(c)
        if cfg.policy == "Random":
            return RandomPolicy(c, seed=cfg.seed)
        raise ValueError(f"Unknown policy '{cfg.policy}'. Implemented: {', '.join(POLICY_INFO)}")

    def draft(self, question, passages):
        if self.draft_fn:
            return self.draft_fn(question, passages)
        from webapp.generation import generate_draft
        return generate_draft(question, passages)


SOURCES = {
    "wikipedia": {"label": "Wikipedia (live API)", "kind": "wikipedia"},
    "hotpotqa": {"label": "HotpotQA Wikipedia abstracts (local Qdrant)", "kind": "qdrant", "collection": "hotpotqa_bge-large-en-v1.5"},
    "asqa": {"label": "ASQA/ALCE Wikipedia corpus (local Qdrant)", "kind": "qdrant", "collection": "alce_asqa_eval_bge-large-en-v1.5"},
}


def stage(emit, sid, status, t0=None, data=None):
    emit({"type": "stage", "stage": sid, "status": status, "duration_s": r(time.monotonic() - t0) if t0 is not None and status != "running" else None, "data": data or {}})


def extract_claims(pipe, draft: str) -> List[ClaimRecord]:
    """Same extraction as RAGVerificationPipeline.run (filter -> decompose, with its whole-draft fallback)."""
    sents = pipe.content_filter.process(draft)
    claims: List[ClaimRecord] = []
    for sent in sents:
        if sent.is_content_bearing:
            claims.extend(pipe.decomposer.decompose(sent))
    if not claims and draft.strip():
        claims = [ClaimRecord(claim_id="c_0", source_sentence_id="s_0", original_text=draft.strip(), normalized_text=draft.strip(), claim_type=ClaimType.FACTUAL_RELATION,
                              decomposition_status=DecompositionStatus.UNSPLIT_PRESERVED, state=ClaimState.PENDING)]
    return claims


def run_pipeline(backend: Backend, cfg: RunConfig, emit: Callable[[Dict[str, Any]], None], cancel: Optional[threading.Event] = None) -> Optional[Dict[str, Any]]:
    cancel = cancel or threading.Event()
    run_t0 = time.monotonic()
    ctx: Dict[str, Any] = {"repaired": set()}
    q = cfg.question.strip()
    try:
        if not q:
            raise ValueError("The question is empty.")
        if cfg.budget < 1 or cfg.budget > 16:
            raise ValueError("Budget must be between 1 and 16 verification calls.")
        if cfg.policy not in POLICY_INFO:
            raise ValueError(f"Unknown policy '{cfg.policy}'. Implemented: {', '.join(POLICY_INFO)}")
        if cfg.source not in SOURCES:
            raise ValueError(f"Unknown source '{cfg.source}'.")
        if cfg.answer_rule not in ANSWER_RULES:
            raise ValueError(f"Unknown answer rule '{cfg.answer_rule}'.")
        emit({"type": "start", "config": dataclasses.asdict(cfg), "policy_info": POLICY_INFO[cfg.policy], "answer_rule": ANSWER_RULES[cfg.answer_rule], "stages": STAGES})
        stage(emit, "question", "done", data={"question": q})
        pipe = backend.pipeline()
        analysis = analyse_question(q)

        # ---- preparation (retrieval, draft, claims, CTS): shared across runs of the same (question, source) so comparisons are paired
        key = (q, cfg.source)
        prep = backend.prep_cache.get(key) if cfg.reuse_prep else None
        if prep is not None:
            for sid in ("retrieval", "draft", "decomposition", "scoring"):
                emit({"type": "stage", "stage": sid, "status": "done", "duration_s": prep["durations"][sid],
                      "data": {**prep["stage_data"][sid], "reused": True, "note": "reused from an earlier run of this question and source (identical draft for a paired comparison); duration is the original"}})
            passages, draft, claims_proto, prep_cost, off_topic = prep["passages"], prep["draft"], prep["claims"], prep["cost"], prep["off_topic"]
        else:
            dummy = ClaimRecord(claim_id="init_query", source_sentence_id="s0", original_text=q, normalized_text=q, claim_type=ClaimType.FACTUAL_RELATION, decomposition_status=DecompositionStatus.ATOMIC)
            stage(emit, "retrieval", "running")
            t0 = time.monotonic()
            retr = backend.retriever(cfg.source)
            try:
                passages, ret_cost = retr.retrieve(dummy, top_k=cfg.top_k)
            except Exception as e:
                stage(emit, "retrieval", "error", t0, {"error": f"{type(e).__name__}: {e}"})
                raise RuntimeError(f"Retrieval failed: {e}. Check the selected source ({SOURCES[cfg.source]['label']}): Wikipedia needs internet access; Qdrant sources need Qdrant on port 6333.") from e
            d_ret = time.monotonic() - t0
            n_retrieved = len(passages)
            gate = passages_support_question(analysis, passages)  # relevance safeguard, any source
            passages = gate["kept"]
            sd_ret = {"passages": [passage_dict(p) for p in passages], "n": len(passages), "n_retrieved": n_retrieved, "dropped_by_relevance_gate": n_retrieved - len(passages),
                      "source": SOURCES[cfg.source]["label"], "analysis": analysis.to_dict(), "trace": getattr(retr, "last_trace", None)}
            stage(emit, "retrieval", "done", t0, sd_ret)
            if not passages:
                tr = getattr(retr, "last_trace", None) or {}
                rejected = len(tr.get("dropped_pages") or [])
                what = " / ".join(analysis.entity_phrases) or "the question's key terms"
                if n_retrieved:
                    msg = gate["reason"]
                elif rejected:
                    msg = f"The search returned {rejected} page{'s' if rejected != 1 else ''}, but none of them is about {what}."
                else:
                    msg = f"The search found no page about {what}."
                raise InsufficientEvidence(msg, {"stage": "retrieval", "analysis": analysis.to_dict(), "trace": tr})
            stage(emit, "draft", "running")
            t0 = time.monotonic()
            try:
                draft, gen_cost = backend.draft(q, passages)
            except Exception as e:
                stage(emit, "draft", "error", t0, {"error": f"{type(e).__name__}: {e}"})
                raise RuntimeError(f"Draft generation failed: {e}. Is the llama-server running (see README)?") from e
            if not draft.strip():
                stage(emit, "draft", "error", t0, {"error": "empty draft"})
                raise RuntimeError("The generator returned an empty draft.")
            draft_source = "generated by the model"
            if not is_insufficient(draft) and not is_well_formed(draft):
                fallback = extractive_draft(analysis, passages)
                if fallback:
                    draft, draft_source = fallback, "extractive fallback: the model returned fragments, so the passage sentences that best answer the question are used verbatim"
            d_draft = time.monotonic() - t0
            sd_draft = {"draft": draft, "tokens": gen_cost.total_tokens, "model": gen_cost.model_id, "draft_source": draft_source, "model_calls": gen_cost.model_calls}
            stage(emit, "draft", "done", t0, sd_draft)
            if is_insufficient(draft):
                raise InsufficientEvidence("The generator reported that the retrieved passages do not contain the answer.",
                                           {"stage": "draft", "analysis": analysis.to_dict(), "trace": getattr(retr, "last_trace", None)})
            stage(emit, "decomposition", "running")
            t0 = time.monotonic()
            all_claims = extract_claims(pipe, draft)
            claims_proto, off_topic = [], []
            for i, c in enumerate(all_claims):  # claims that share no meaningful word with the question are reported, not verified or cited
                if claim_is_on_topic(analysis, c.normalized_text, is_first=(i == 0)):
                    claims_proto.append(c)
                else:
                    off_topic.append({"claim_id": c.claim_id, "text": c.normalized_text, "reason": "shares no meaningful word with the question"})
            d_dec = time.monotonic() - t0
            sd_dec = {"claims": [{"claim_id": c.claim_id, "text": c.normalized_text, "source_sentence": c.original_text, "claim_type": c.claim_type.value,
                                  "decomposition": c.decomposition_status.value} for c in claims_proto], "off_topic": off_topic, "n_in_draft": len(all_claims)}
            stage(emit, "decomposition", "done", t0, sd_dec)
            stage(emit, "scoring", "running")
            t0 = time.monotonic()
            for c in claims_proto:
                c.triage_score = pipe.triage_scorer.score_claim(c, context_text=draft)
            d_sc = time.monotonic() - t0
            sd_sc = {"claims": [{"claim_id": c.claim_id, "text": c.normalized_text, "cts": r(c.triage_score), "claim_type": c.claim_type.value,
                                 "cts_components": pipe.triage_scorer.compute_features(c, context_text=draft)} for c in claims_proto]}
            stage(emit, "scoring", "done", t0, sd_sc)
            prep_cost = {"tokens": gen_cost.total_tokens or 0, "latency_s": (ret_cost.latency_seconds or 0) + (gen_cost.latency_seconds or 0)}
            if cfg.reuse_prep:
                backend.prep_cache[key] = {"passages": passages, "draft": draft, "claims": claims_proto, "cost": prep_cost, "off_topic": off_topic,
                                           "durations": {"retrieval": r(d_ret), "draft": r(d_draft), "decomposition": r(d_dec), "scoring": r(d_sc)},
                                           "stage_data": {"retrieval": sd_ret, "draft": sd_draft, "decomposition": sd_dec, "scoring": sd_sc}}
        if not claims_proto:
            raise RuntimeError("No verifiable claims were extracted from the draft.")

        # ---- controller loop (stages 6-9 are interleaved per round; the stage cards summarise them)
        claims = [ClaimRecord(claim_id=c.claim_id, source_sentence_id=c.source_sentence_id, original_text=c.original_text, normalized_text=c.normalized_text, claim_type=c.claim_type,
                              decomposition_status=c.decomposition_status, state=ClaimState.PENDING, triage_score=c.triage_score) for c in claims_proto]
        store = {c.claim_id: list(passages) for c in claims}
        policy = ObservedPolicy(backend.make_policy(cfg), cfg.policy, emit, ctx, cancel)
        inner_assessor = backend.assessor(cfg.verifier)
        ctl = VerificationController(retriever=ObservedRetriever(backend.retriever(cfg.source), emit, cancel), assessor=ObservedAssessor(inner_assessor, backend.memo, emit, ctx, cancel),
                                     repairer=ObservedRepairer(backend.repairer(), backend.memo, emit, ctx, cancel), policy=policy,
                                     global_budget=VerificationBudget(max_steps=50, max_model_calls=cfg.budget), per_claim_budget=PerClaimBudget())
        for sid in ("selection", "verification", "feedback"):
            stage(emit, sid, "running")
        emit({"type": "loop", "phase": "start", "budget": cfg.budget, "claims": len(claims)})
        t0 = time.monotonic()
        trace = ctl.run(claims=claims, initial_evidence=store, run_id=f"web_{cfg.policy}_{cfg.budget}")
        loop_s = time.monotonic() - t0
        out = claim_outcomes(claims, trace)
        n_rounds = policy.round
        n_verify = sum(1 for s in trace.steps if s.action == ControllerAction.VERIFY)
        n_repair = sum(1 for s in trace.steps if s.action == ControllerAction.REPAIR)
        for sid in ("selection", "verification", "feedback"):
            stage(emit, sid, "done", t0, {"rounds": n_rounds, "verifications": n_verify})
        if n_repair:
            stage(emit, "repair", "done", t0, {"repairs": n_repair, "re_verified_pass": len(out["repairs_pass"])})
        else:
            stage(emit, "repair", "skipped", None, {"reason": "no claim was contradicted, or no budget remained for a repair"})
        calls_used = trace.total_cost.get("model_calls", 0)
        emit({"type": "budget_check", "configured": cfg.budget, "model_calls_used": calls_used, "within_budget": calls_used <= cfg.budget, "stopping_reason": trace.stopping_reason.value,
              "steps": trace.total_steps, "feedback_failures": ctl.feedback_failures})

        # ---- answer reconstruction
        stage(emit, "answer", "running")
        t1 = time.monotonic()
        answer, items = assemble(cfg.answer_rule, claims, out, store)
        by_id = {c.claim_id: c for c in claims}
        title_url: Dict[str, Tuple[Optional[str], str]] = {}
        for plist in store.values():
            for p in plist:
                title_url.setdefault(p.doc_title, passage_url(p))
        in_answer = {it["claim_id"] for it in items}
        ledger = []
        for c in claims:
            code, why = claim_status(c.claim_id, out, by_id)
            ra = out["repairs_all"].get(c.claim_id)
            sup = support_for_claim(c.normalized_text, store.get(c.claim_id, []))
            ledger.append({"claim_id": c.claim_id, "text": c.normalized_text, "claim_type": c.claim_type.value, "cts": r(c.triage_score), "status": code, "explanation": why,
                           "verdict": out["last"].get(c.claim_id), "in_final_answer": c.claim_id in in_answer, "checked": c.claim_id in out["last"],
                           "repair": {"revision": ra.get("proposed_revision"), "reason": ra.get("repair_reason"), "reverification": ra.get("reverification")} if ra else None,
                           "lexical_support_title": sup.get("title"),
                           "lexical_support_note": "lexical proxy (token overlap), not a truth label"})
        cited = [{"claim_id": it["claim_id"], "text": it["text"], "title": it.get("title"), "url": (title_url.get(it.get("title")) or (None, ""))[0],
                  "url_provenance": (title_url.get(it.get("title")) or (None, ""))[1], "status": next(l["status"] for l in ledger if l["claim_id"] == it["claim_id"])} for it in items]
        stage(emit, "answer", "done", t1, {"answer": answer, "rule": cfg.answer_rule})
        checked = sum(1 for l in ledger if l["checked"])
        verif_cost = trace.total_cost
        summary = {
            "config": dataclasses.asdict(cfg), "draft": draft, "off_topic_claims": off_topic,
            "sources": [{"title": t, "url": u} for t, (u, _) in sorted({p.doc_title: passage_url(p) for p in passages}.items())], "answer": answer, "answer_items": cited, "answer_rule_text": ANSWER_RULES[cfg.answer_rule], "ledger": ledger,
            "metrics": {
                "sources_retrieved": len({p.doc_title for p in passages}), "passages_retrieved": len(passages), "claims_off_topic": len(off_topic),
                "budget_configured": cfg.budget, "model_calls_used": calls_used, "within_budget": calls_used <= cfg.budget, "claims_available": len(claims), "claims_checked": checked,
                "claims_retained": len(in_answer), "claims_excluded": len(claims) - len(in_answer),
                "claim_retention": r(len(in_answer) / len(claims)), "repairs_attempted": len(out["repairs_all"]), "repairs_reverified_pass": len(out["repairs_pass"]),
                "unverified_claims": sum(1 for l in ledger if l["status"] == "unverified"), "insufficient_claims": sum(1 for l in ledger if l["status"] == "insufficient"),
                "contradicted_claims": sum(1 for l in ledger if l["status"] in ("contradicted", "repair_failed")),
                "retained_insufficient": sum(1 for l in ledger if l["status"] == "insufficient" and l["in_final_answer"]),
                "retained_lexically_unsupported": sum(1 for l in ledger if l["in_final_answer"] and l["lexical_support_title"] is None),
                "verification_tokens": verif_cost.get("total_tokens", 0), "verification_recorded_latency_s": r(verif_cost.get("latency_seconds", 0.0)),
                "loop_wall_s": r(loop_s), "total_wall_s": r(time.monotonic() - run_t0), "stopping_reason": trace.stopping_reason.value, "rounds": n_rounds,
                "policy": cfg.policy, "verifier": inner_assessor.assessor_id,
            },
            "metric_definitions": {"claim_retention": "claims in the final answer / claims in the draft (a coverage measure; NOT factuality, NOT STR-EM)",
                                   "retained_lexically_unsupported": "retained claims with no lexically supporting passage: a proxy, not a truth label",
                                   "model_calls_used": "calls counted by the controller's BudgetTracker (verification + repair + re-verification + any retrieval that costs a model call)"},
            "not_measured": ["factual correctness / real-world truth (needs human annotation)", "support rates of the answer (see docs/alce_support_annotation_report.md: annotation pending)"],
        }
        emit({"type": "result", "result": summary})
        return summary
    except RunCancelled:
        emit({"type": "cancelled"})
        return None
    except InsufficientEvidence as e:
        emit({"type": "insufficient_evidence", "message": str(e), **e.data})
        return None
    except Exception as e:
        emit({"type": "error", "message": str(e), "error_type": type(e).__name__, "trace": traceback.format_exc(limit=3)})
        return None
