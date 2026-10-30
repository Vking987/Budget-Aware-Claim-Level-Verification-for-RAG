"""Tests for webapp/ (Adaptive RAG Lab): real runner + controller + policies with scripted retriever/assessor/repairer (no network, no models)."""

import json
import sys
import threading
import unittest
from pathlib import Path

from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from src.controller.baselines import RandomPolicy  # noqa: E402
from src.controller.interfaces import MockRepairer, MockRetriever  # noqa: E402
from src.controller.mab_policy import ContextualMABPolicy  # noqa: E402
from src.controller.models import RepairResult  # noqa: E402
from src.controller.policy import StaticAllocationPolicy, StaticCTSRankedPolicy  # noqa: E402
from src.evidence.assessor import MockDeterministicAssessor  # noqa: E402
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost  # noqa: E402
from webapp import runner as W  # noqa: E402
from webapp.server import make_app  # noqa: E402
from webapp.wikipedia_retriever import WikipediaError, WikipediaRetriever, rank_passages, split_passages  # noqa: E402

QUESTION = "What is the novel Frankenstein about?"  # an explanatory question: claims that name the entity or the novel are on topic
DRAFT = ("Frankenstein is a novel written by Mary Shelley. Frankenstein was first published in 1818. The novel is set in Geneva in 1950. "
         "The author of Frankenstein, Mary Shelley, was born in London. In the novel the monster is called Victor. The novel was adapted into a film in 1931.")
PASSAGES = [EvidencePassage(evidence_id="p1", doc_title="Frankenstein", source_text="Frankenstein is an 1818 novel written by Mary Shelley, first published in 1818.", retrieval_score=0.9,
                            metadata={"url": "https://en.wikipedia.org/wiki/Frankenstein"}),
            EvidencePassage(evidence_id="p2", doc_title="Mary Shelley", source_text="Mary Shelley was born in London in 1797.", retrieval_score=0.5, metadata={"url": "https://en.wikipedia.org/wiki/Mary_Shelley"})]


class Ret(MockRetriever):
    def __init__(self, fail=False, empty=False):
        self.fail, self.empty = fail, empty

    def retrieve(self, claim, top_k=3):
        if self.fail:
            raise ConnectionError("network down")
        return ([] if self.empty else list(PASSAGES)), VerificationCost(model_calls=0)


class Assessor(MockDeterministicAssessor):
    """geneva/victor -> CONTRADICTED; film -> INSUFFICIENT; everything else SUPPORTED. One model call each."""

    @property
    def assessor_id(self):
        return "scripted_assessor"

    def assess(self, claim, passages):
        t = claim.normalized_text.lower()
        lab = AssessmentLabel.CONTRADICTED if ("geneva" in t or "victor" in t) else AssessmentLabel.INSUFFICIENT if "film" in t else AssessmentLabel.SUPPORTED
        return EvidenceAssessment(claim_id=claim.claim_id, evidence_ids=[p.evidence_id for p in passages], supporting_spans={"p1": ["span text"]} if lab == AssessmentLabel.SUPPORTED else {},
                                  label=lab, confidence=0.9, rationale="scripted", cost=VerificationCost(model_calls=1, latency_seconds=0.01, total_tokens=5))


class Repairer(MockRepairer):
    def repair(self, claim, assessment, passages):
        return RepairResult(claim_id=claim.claim_id, original_text=claim.normalized_text, proposed_revision="Mary Shelley wrote Frankenstein.", repair_reason="scripted",
                            supporting_evidence_ids=["p1"], confidence=0.9, cost=VerificationCost(model_calls=1)), VerificationCost(model_calls=1)


def backend(**kw):
    kw.setdefault("retriever_factory", lambda s: Ret())
    kw.setdefault("assessor_factory", lambda n: Assessor())
    kw.setdefault("repairer", Repairer())
    kw.setdefault("draft_fn", lambda q, ps: (DRAFT, VerificationCost(model_calls=1, total_tokens=10, model_id="draft_model")))
    return W.Backend(**kw)


def run(b, **cfg):
    ev = []
    res = W.run_pipeline(b, W.RunConfig(question=cfg.pop("question", QUESTION), **cfg), ev.append)
    return ev, res


class TestPipelineEvents(unittest.TestCase):
    def test_stage_events_are_emitted_in_pipeline_order_from_real_execution(self):
        ev, res = run(backend(), budget=4, policy="StaticFIFO")
        seq = [e["stage"] for e in ev if e["type"] == "stage" and e["status"] == "done"]
        for earlier, later in (("question", "retrieval"), ("retrieval", "draft"), ("draft", "decomposition"), ("decomposition", "scoring"), ("scoring", "selection"), ("selection", "answer")):
            self.assertLess(seq.index(earlier), seq.index(later))
        types = [e["type"] for e in ev]
        self.assertLess(types.index("round"), types.index("verify"))
        self.assertLess(types.index("verify"), types.index("result"))
        self.assertEqual(types[-1], "result")
        self.assertIsNotNone(res)

    def test_decomposition_matches_the_draft_and_cts_is_attached(self):
        ev, _ = run(backend(), budget=2)
        sc = next(e for e in ev if e["type"] == "stage" and e["stage"] == "scoring" and e["status"] == "done")
        self.assertEqual(len(sc["data"]["claims"]), 6)
        self.assertTrue(all(0.0 <= c["cts"] <= 1.0 and set(c["cts_components"]) == {"type_weight", "specificity", "novelty", "complexity"} for c in sc["data"]["claims"]))

    def test_second_run_reuses_the_same_draft_for_paired_comparison(self):
        b = backend()
        run(b, budget=2)
        ev, _ = run(b, budget=4)
        self.assertTrue(all(e["data"].get("reused") for e in ev if e["type"] == "stage" and e["stage"] in ("retrieval", "draft", "decomposition", "scoring")))


class TestBudgetEnforcement(unittest.TestCase):
    def test_low_medium_high_budgets_are_never_exceeded_and_are_used(self):
        used = {}
        for b in (2, 4, 8):
            ev, res = run(backend(), budget=b, policy="StaticCTSRanked")
            chk = next(e for e in ev if e["type"] == "budget_check")
            self.assertTrue(chk["within_budget"])
            self.assertLessEqual(chk["model_calls_used"], b)
            self.assertEqual(res["metrics"]["model_calls_used"], chk["model_calls_used"])
            used[b] = chk["model_calls_used"]
        self.assertEqual(used[2], 2)
        self.assertGreater(used[4], used[2])
        self.assertGreaterEqual(used[8], used[4])

    def test_unchecked_claims_are_reported_as_unverified_when_the_budget_ends(self):
        _, res = run(backend(), budget=2, policy="StaticFIFO")
        m = res["metrics"]
        self.assertGreater(m["unverified_claims"], 0)
        self.assertEqual(m["claims_checked"] + m["unverified_claims"], m["claims_available"])
        self.assertTrue(any(l["status"] == "unverified" for l in res["ledger"]))

    def test_invalid_budget_is_rejected_before_any_work(self):
        ev, res = run(backend(), budget=0)
        self.assertIsNone(res)
        self.assertEqual(ev[-1]["type"], "error")
        self.assertFalse(any(e["type"] == "stage" for e in ev))


class TestPolicySelection(unittest.TestCase):
    def test_selector_builds_the_selected_policy_class(self):
        b = W.Backend()
        for name, cls in (("LinUCB", ContextualMABPolicy), ("StaticCTSRanked", StaticCTSRankedPolicy), ("StaticFIFO", StaticAllocationPolicy), ("Random", RandomPolicy)):
            self.assertIsInstance(b.make_policy(W.RunConfig(question="q", policy=name)), cls)
        self.assertTrue(b.make_policy(W.RunConfig(question="q", policy="LinUCB")).frozen)
        with self.assertRaises(ValueError):
            b.make_policy(W.RunConfig(question="q", policy="TypeThompson"))

    def test_rounds_report_the_selected_policy_and_unknown_policy_is_an_error(self):
        for name in W.POLICY_INFO:
            ev, _ = run(backend(), budget=2, policy=name)
            self.assertTrue(all(e["policy"] == name for e in ev if e["type"] == "round"))
        ev, res = run(backend(), budget=2, policy="TypeThompson")
        self.assertIsNone(res)
        self.assertIn("Unknown policy", ev[-1]["message"])

    def test_linucb_round_scores_are_the_policys_own_and_match_what_ran(self):
        ev, _ = run(backend(), budget=4, policy="LinUCB")
        rounds = [e for e in ev if e["type"] == "round" and e["chosen"]["action"] != "STOP"]
        self.assertTrue(rounds)
        for e in rounds:
            top = e["candidates"][0]
            self.assertEqual((top["claim_id"], top["action"]), (e["chosen"]["claim_id"], e["chosen"]["action"]))
            for c in e["candidates"]:
                self.assertAlmostEqual(c["score"], c["mean"] + c["bonus"], places=3)
                self.assertEqual(len(c["features"]), 8)
            self.assertTrue(e["frozen"])
        verified = [e["claim_id"] for e in ev if e["type"] == "verify" and e["phase"] == "start" and not e["reverification"]]
        chosen_verify = [e["chosen"]["claim_id"] for e in rounds if e["chosen"]["action"] == "VERIFY"]
        self.assertEqual(verified, chosen_verify)

    def test_stop_rounds_list_no_invented_candidates_for_any_policy(self):
        for name in W.POLICY_INFO:
            ev, _ = run(backend(), budget=2, policy=name)
            stops = [e for e in ev if e["type"] == "round" and e["chosen"]["action"] == "STOP"]
            self.assertTrue(stops)
            self.assertTrue(all(e["candidates"] == [] for e in stops))

    def test_feedback_never_claims_a_policy_update_for_a_frozen_or_static_policy(self):
        for name in ("LinUCB", "StaticFIFO"):
            ev, _ = run(backend(), budget=2, policy=name)
            fb = [e for e in ev if e["type"] == "feedback"]
            self.assertTrue(fb)
            self.assertTrue(all(e["policy_updated"] is False for e in fb))
            if name == "StaticFIFO":
                self.assertTrue(all(e["reward"] is None and e["policy_learns"] is False for e in fb))

    def test_static_policies_expose_their_real_ranking_key(self):
        ev, _ = run(backend(), budget=2, policy="StaticCTSRanked")
        r1 = next(e for e in ev if e["type"] == "round")
        scores = [c["score"] for c in r1["candidates"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        ev, _ = run(backend(), budget=2, policy="Random")
        self.assertTrue(all(c["score"] is None for e in ev if e["type"] == "round" for c in e["candidates"]))


class TestAnswerReconstruction(unittest.TestCase):
    def test_rules_differ_exactly_in_insufficient_claims_and_repairs(self):
        r = {k: run(backend(), budget=16, policy="StaticFIFO", answer_rule=k)[1] for k in ("S0", "S1", "S2")}
        led = lambda k: {l["claim_id"]: l for l in r[k]["ledger"]}  # noqa: E731
        ins = [c for c, l in led("S1").items() if l["status"] == "insufficient"]
        self.assertTrue(ins)
        for c in ins:
            self.assertFalse(led("S0")[c]["in_final_answer"])
            self.assertTrue(led("S1")[c]["in_final_answer"])
        self.assertEqual(r["S1"]["metrics"]["retained_insufficient"], len(ins))
        self.assertEqual(r["S0"]["metrics"]["retained_insufficient"], 0)
        for c, l in led("S1").items():
            if l["verdict"] == "CONTRADICTED":
                self.assertFalse(l["in_final_answer"])

    def test_assembly_matches_the_replay_study_code(self):
        import replay_answer_handling as P
        from src.claims.models import ClaimRecord, ClaimType, DecompositionStatus
        from src.controller.budget import PerClaimBudget, VerificationBudget
        from src.controller.controller import VerificationController
        from src.controller.policy import ControllerPolicyConfig
        texts = ["Mary Shelley wrote Frankenstein.", "The novel is set in Geneva in 1950.", "The book was adapted into a film in 1931."]
        claims = [ClaimRecord(claim_id=f"c{i}", source_sentence_id=f"s{i}", original_text=t, normalized_text=t, claim_type=ClaimType.FACTUAL_RELATION,
                              decomposition_status=DecompositionStatus.ATOMIC, triage_score=0.5) for i, t in enumerate(texts)]
        store = {c.claim_id: list(PASSAGES) for c in claims}
        trace = VerificationController(retriever=Ret(), assessor=Assessor(), repairer=Repairer(), policy=StaticAllocationPolicy(ControllerPolicyConfig(enable_repair=True)),
                                       global_budget=VerificationBudget(max_steps=50, max_model_calls=16), per_claim_budget=PerClaimBudget()).run(claims=claims, initial_evidence=store, run_id="t")
        out = W.claim_outcomes(claims, trace)
        for s, rule in (("S0_current", "S0"), ("S1_conservative", "S1"), ("S2_repair_first", "S2")):
            self.assertEqual(P.assemble(s, claims, P.claim_outcomes(claims, trace), store, [])["answer"], W.assemble(rule, claims, out, store)[0])


class TestCitations(unittest.TestCase):
    def test_every_cited_url_comes_from_a_retrieved_passage(self):
        _, res = run(backend(), budget=16, policy="StaticFIFO")
        retrieved = {p.metadata["url"] for p in PASSAGES}
        self.assertTrue(res["answer_items"])
        for it in res["answer_items"]:
            self.assertIn(it["url"], retrieved)
            self.assertEqual(it["url_provenance"], "source")
            self.assertIn(f"[{it['title']}]", res["answer"])

    def test_url_is_derived_and_flagged_when_a_passage_has_none(self):
        p = EvidencePassage(evidence_id="x", doc_title="Mary Shelley", source_text="t", metadata={})
        self.assertEqual(W.passage_url(p), ("https://en.wikipedia.org/wiki/Mary_Shelley", "derived from title"))
        self.assertEqual(W.passage_url(EvidencePassage(evidence_id="y", doc_title="Untitled", source_text="t")), (None, "none"))


class TestFailures(unittest.TestCase):
    def test_retrieval_failure_is_a_clean_error_with_no_result(self):
        ev, res = run(backend(retriever_factory=lambda s: Ret(fail=True)))
        self.assertIsNone(res)
        self.assertIn("Retrieval failed", ev[-1]["message"])
        self.assertTrue(any(e["type"] == "stage" and e["stage"] == "retrieval" and e["status"] == "error" for e in ev))
        self.assertFalse(any(e["type"] == "result" for e in ev))

    def test_empty_retrieval_never_produces_an_answer(self):
        ev, res = run(backend(retriever_factory=lambda s: Ret(empty=True)))
        self.assertIsNone(res)
        self.assertEqual(ev[-1]["type"], "insufficient_evidence")
        self.assertIn("found no page", ev[-1]["message"])
        self.assertFalse(any(e["type"] == "stage" and e["stage"] == "draft" for e in ev))  # no draft is generated without evidence

    def test_model_server_down_is_reported_for_the_draft(self):
        def boom(q, ps):
            raise RuntimeError("connection refused")
        ev, res = run(backend(draft_fn=boom))
        self.assertIsNone(res)
        self.assertIn("Draft generation failed", ev[-1]["message"])
        self.assertIn("llama-server", ev[-1]["message"])

    def test_cancel_stops_between_controller_steps(self):
        cancel = threading.Event()
        ev = []

        def emit(e):
            ev.append(e)
            if e["type"] == "verify" and e["phase"] == "done":
                cancel.set()
        W.run_pipeline(backend(), W.RunConfig(question=QUESTION, budget=8, policy="StaticFIFO"), emit, cancel)
        self.assertEqual(ev[-1]["type"], "cancelled")
        self.assertEqual(sum(1 for e in ev if e["type"] == "verify" and e["phase"] == "done"), 1)


class TestServer(unittest.TestCase):
    HEALTH = {"llama_server": {"ok": True, "needed_for": ""}, "qdrant": {"ok": True, "collections": ["hotpotqa_bge-large-en-v1.5"], "needed_for": ""},
              "wikipedia_api": {"ok": False, "needed_for": ""}, "nli_model": {"ok": True, "needed_for": ""}, "linucb_checkpoint": {"ok": True, "needed_for": ""}}

    def client(self, b=None):
        return TestClient(make_app(b or backend(), health_fn=lambda: self.HEALTH))

    def test_config_lists_only_implemented_policies_and_marks_unavailable_sources(self):
        cfg = self.client().get("/api/config").json()
        self.assertEqual({p["id"] for p in cfg["policies"]}, {"LinUCB", "StaticCTSRanked", "StaticFIFO", "Random"})
        self.assertEqual(cfg["budgets"], {"low": 2, "medium": 4, "high": 8})
        src = {s["id"]: s["available"] for s in cfg["sources"]}
        self.assertFalse(src["wikipedia"])
        self.assertTrue(src["hotpotqa"])
        self.assertFalse(src["asqa"])
        self.assertIn("TypeThompson", [u["id"] for u in cfg["unavailable_by_design"]])

    def test_sse_stream_carries_the_events_and_ends_with_result(self):
        with self.client().stream("POST", "/api/run", json={"question": "Who wrote Frankenstein?", "budget": 2, "policy": "StaticFIFO", "verifier": "nli"}) as r:
            self.assertEqual(r.headers["content-type"].split(";")[0], "text/event-stream")
            events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual([e["seq"] for e in events], sorted(e["seq"] for e in events))
        self.assertEqual(events[-1]["result"]["metrics"]["budget_configured"], 2)

    def test_busy_server_and_bad_request(self):
        b = backend()
        c = self.client(b)
        b.lock.acquire()
        try:
            self.assertEqual(c.post("/api/run", json={"question": "q"}).status_code, 409)
        finally:
            b.lock.release()
        self.assertEqual(c.post("/api/run", content=b"not json").status_code, 400)

    def test_index_is_served(self):
        r = self.client().get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Adaptive RAG Lab", r.text)


class TestWikipediaRetriever(unittest.TestCase):
    def test_passage_split_and_overlap_ranking(self):
        ps = split_passages("Mary Shelley wrote Frankenstein in 1818. " * 30, 200)
        self.assertTrue(all(len(p) <= 200 for p in ps))
        ranked = rank_passages("who wrote frankenstein", [{"title": "A", "url": "u", "text": "Cats purr.", "rank": 0}, {"title": "B", "url": "v", "text": "Mary Shelley wrote Frankenstein.", "rank": 1}])
        self.assertEqual(ranked[0]["title"], "B")
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])

    def test_network_failure_raises_and_empty_search_is_empty(self):
        class Sess:
            def __init__(self, payload=None, exc=None):
                self.payload, self.exc = payload, exc

            def get(self, url, params=None, timeout=None):
                if self.exc:
                    raise self.exc
                payload = self.payload

                class R:
                    def raise_for_status(self):
                        pass

                    def json(self):
                        return payload
                return R()
        from src.claims.models import ClaimRecord, ClaimType, DecompositionStatus
        c = ClaimRecord(claim_id="q", source_sentence_id="s", original_text="x y", normalized_text="x y", claim_type=ClaimType.FACTUAL_RELATION, decomposition_status=DecompositionStatus.ATOMIC)
        with self.assertRaises(WikipediaError):
            WikipediaRetriever(session=Sess(exc=TimeoutError("t"))).retrieve(c)
        ps, _ = WikipediaRetriever(session=Sess({"query": {"search": []}})).retrieve(c)
        self.assertEqual(ps, [])


if __name__ == "__main__":
    unittest.main()
