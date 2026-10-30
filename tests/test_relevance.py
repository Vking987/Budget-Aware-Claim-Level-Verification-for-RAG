"""Regression tests for retrieval relevance and question-to-answer alignment (webapp/relevance.py, wikipedia_retriever.py, generation.py, runner gates).

A scripted MediaWiki reproduces the failure seen with "Who created VIT University and when?": a full-question search returns Balut, Anna University
and Hardbag, and the right page (Vellore Institute of Technology) is not among the hits. No network, no model.
"""

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.claims.models import ClaimRecord, ClaimType, DecompositionStatus  # noqa: E402
from src.evidence.models import EvidencePassage, VerificationCost  # noqa: E402
from webapp import runner as W  # noqa: E402
from webapp.generation import INSUFFICIENT, build_prompt, clean_draft, extractive_draft, generate_draft, is_insufficient, is_well_formed  # noqa: E402
from webapp.relevance import (acronym, analyse_question, claim_is_on_topic, entity_coverage, mentions_entity, passages_support_question, title_match)  # noqa: E402
from webapp.wikipedia_retriever import WikipediaRetriever  # noqa: E402

import test_webapp as TW  # noqa: E402  (scripted assessor / repairer / retriever helpers)

PAGES = {
    "Balut (food)": "Balut is a fertilized developing egg embryo that is boiled and eaten from the shell.",
    "Anna University": "Anna University is a public state university located in Chennai, Tamil Nadu, India. The main campus is in Guindy.",
    "Hardbag": "Hardbag is a subgenre of hard house music. The popularity of the hardbag genre reached its zenith in 1995 when it was created in the clubs.",
    "Vellore Institute of Technology": "Vellore Institute of Technology (VIT) is a private deemed university in Vellore, Tamil Nadu, India. The institution was founded in 1984 "
                                       "by G. Viswanathan as Vellore Engineering College. It was established as a deemed university in 2001.",
    "VIT Bhopal University": "VIT Bhopal University (VIT B) is a private university in Madhya Pradesh. It was established in 2017 by the Vellore group.",
    "Vidyalankar Institute of Technology": "Vidyalankar Institute of Technology is an engineering college in Mumbai. It was established in 1999.",
    "VIT": "VIT may refer to: Vellore Institute of Technology, Vision transformer, Vitamin.",
    "Frankenstein": "Frankenstein; or, The Modern Prometheus is an 1818 Gothic novel written by English author Mary Shelley. It was first published anonymously in London in 1818.",
    "Frankenstein (1931 film)": "Frankenstein is a 1931 American horror film directed by James Whale.",
}
BODY = {"Vellore Institute of Technology": "The university runs a festival every February. It was ranked 801-1000 by Times Higher Education in 2023. Students protested hostel food."}
SEARCH = {  # query text (lower case) -> hit order; mirrors what the live MediaWiki search returned
    "created": ["Balut (food)", "Anna University", "Hardbag", "Vellore Institute of Technology"],
    "vit university": ["Vellore Institute of Technology", "VIT Bhopal University", "VIT", "Anna University", "Vidyalankar Institute of Technology"],
    "frankenstein": ["Frankenstein", "Frankenstein (1931 film)"],
    "vit": ["VIT", "Vellore Institute of Technology", "VIT Bhopal University"],
}


class FakeWiki:
    """Stands in for requests.Session against the MediaWiki API (generator=search with leads, and full-text extracts by title)."""

    def __init__(self, fail_first: int = 0):
        self.calls, self.fail_first = 0, fail_first

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if self.calls <= self.fail_first:
            return SimpleNamespace(status_code=429, headers={"Retry-After": "0"}, raise_for_status=lambda: (_ for _ in ()).throw(RuntimeError("429")), json=lambda: {})
        if "gsrsearch" in params:
            q = params["gsrsearch"].lower()
            key = next((k for k in ("created", "vit university", "frankenstein", "vit") if (k == "created" and "created" in q) or (k != "created" and q.startswith(k) and (k != "vit" or q == "vit"))), None)
            titles = SEARCH.get(key, [])
            pages = [{"title": t, "index": i + 1, "extract": PAGES[t], "fullurl": "https://en.wikipedia.org/wiki/" + t.replace(" ", "_"),
                      "pageprops": {"disambiguation": ""} if t == "VIT" else {}} for i, t in enumerate(titles)]
        else:
            t = params["titles"]
            pages = [{"title": t, "extract": PAGES[t] + " " + BODY.get(t, "")}]
        payload = {"query": {"pages": pages}}
        return SimpleNamespace(status_code=200, headers={}, raise_for_status=lambda: None, json=lambda: payload)


def retrieve(question, wiki=None, top_k=5):
    r = WikipediaRetriever(session=wiki or FakeWiki(), min_interval=0.0)
    c = ClaimRecord(claim_id="q", source_sentence_id="s", original_text=question, normalized_text=question, claim_type=ClaimType.FACTUAL_RELATION, decomposition_status=DecompositionStatus.ATOMIC)
    ps, _ = r.retrieve(c, top_k=top_k)
    return ps, r.last_trace


class TestQuestionAnalysis(unittest.TestCase):
    def test_entity_phrase_is_extracted_and_question_words_are_not_entities(self):
        cases = {"Who created VIT University and when?": (["VIT University"], ["vit", "university"]),
                 "Who founded VIT and when was it established?": (["VIT"], ["vit"]),
                 "Who wrote Frankenstein and when was it first published?": (["Frankenstein"], ["frankenstein"]),
                 "What caused the fall of the Western Roman Empire?": (["Western Roman Empire"], ["western", "roman", "empire"])}
        for q, (phrases, tokens) in cases.items():
            a = analyse_question(q)
            self.assertEqual(a.entity_phrases, phrases, q)
            self.assertEqual(a.entity_tokens, tokens, q)
            self.assertNotIn("who", a.base_tokens)
            self.assertNotIn("when", a.base_tokens)
        self.assertEqual(analyse_question("Who created VIT University and when?").queries[0], "VIT University")

    def test_intent_words_use_synonyms_but_not_very_common_words(self):
        a = analyse_question("Who created VIT University and when?")
        self.assertEqual(a.intent_tokens, ["creat"])
        self.assertEqual(a.intent_hits("The institution was founded in 1984."), 1)  # created -> founded
        self.assertEqual(a.intent_hits("The campus set a new record and the result was good."), 0)

    def test_acronym_matches_a_title_whose_initials_spell_it(self):
        self.assertEqual(acronym("Vellore Institute of Technology"), "vit")
        self.assertTrue(mentions_entity(["vit", "university"], "Vellore Institute of Technology", PAGES["Vellore Institute of Technology"]))
        self.assertFalse(mentions_entity(["vit", "university"], "Anna University", PAGES["Anna University"]))
        self.assertEqual(entity_coverage(["vit"], "Balut (food)", PAGES["Balut (food)"]), 0.0)

    def test_title_match_prefers_the_exact_entity_over_extended_titles(self):
        a = analyse_question("Who created VIT University and when?")
        tm = lambda t: title_match(a.entity_tokens, t, PAGES[t], a.content_tokens)  # noqa: E731
        self.assertEqual(tm("Vellore Institute of Technology"), 1.0)
        self.assertLess(tm("VIT Bhopal University"), 0.7)
        f = analyse_question("Who wrote Frankenstein and when was it first published?")
        self.assertEqual(title_match(f.entity_tokens, "Frankenstein", PAGES["Frankenstein"], f.content_tokens), 1.0)
        self.assertLess(title_match(f.entity_tokens, "Frankenstein (1931 film)", PAGES["Frankenstein (1931 film)"], f.content_tokens), 0.7)


class TestRetrievalRelevance(unittest.TestCase):
    def test_vit_question_returns_the_right_university_and_drops_the_failure_pages(self):
        ps, tr = retrieve("Who created VIT University and when?")
        titles = {p.doc_title for p in ps}
        self.assertEqual(titles, {"Vellore Institute of Technology"})
        dropped = {d["title"]: d["reason"] for d in tr["dropped_pages"]}
        for bad in ("Balut (food)", "Anna University", "Hardbag"):
            self.assertIn("does not mention", dropped[bad])
        self.assertEqual(dropped["VIT"], "disambiguation page")
        self.assertIn("different entity", dropped["VIT Bhopal University"])
        self.assertIn("Vidyalankar Institute of Technology", dropped)  # another "VIT": its lead never says "university", so the entity gate drops it
        self.assertIn("founded in 1984", ps[0].source_text)  # the passage that answers who/when is first

    def test_acronym_only_question_also_finds_the_acronym_expansion(self):
        ps, tr = retrieve("Who founded VIT and when was it established?")
        self.assertEqual(ps[0].doc_title, "Vellore Institute of Technology")
        self.assertNotIn("Hardbag", {p.doc_title for p in ps})

    def test_entity_questions_ignore_pages_about_a_related_work_or_entity(self):
        ps, tr = retrieve("Who wrote Frankenstein and when was it first published?")
        self.assertEqual({p.doc_title for p in ps}, {"Frankenstein"})
        self.assertIn("different entity", {d["title"]: d["reason"] for d in tr["dropped_pages"]}["Frankenstein (1931 film)"])

    def test_provenance_and_url_are_preserved_and_passages_are_not_duplicated(self):
        ps, _ = retrieve("Who created VIT University and when?")
        self.assertTrue(all(p.metadata["url"].startswith("https://en.wikipedia.org/wiki/") and p.metadata["source"] == "wikipedia" for p in ps))
        keys = [" ".join(p.source_text.lower().split())[:90] for p in ps]
        self.assertEqual(len(keys), len(set(keys)))

    def test_rate_limit_is_retried_with_backoff_then_succeeds(self):
        wiki = FakeWiki(fail_first=2)
        orig = time.sleep
        time.sleep = lambda s: None
        try:
            ps, _ = retrieve("Who wrote Frankenstein and when was it first published?", wiki)
        finally:
            time.sleep = orig
        self.assertTrue(ps)
        self.assertGreater(wiki.calls, 2)


class TestGatesInTheRunner(unittest.TestCase):
    def _run(self, retriever, draft_fn, question="Who created VIT University and when?"):
        ev = []
        b = TW.backend(retriever_factory=lambda s: retriever, draft_fn=draft_fn)
        res = W.run_pipeline(b, W.RunConfig(question=question, budget=4, policy="StaticFIFO"), ev.append)
        return ev, res

    def test_irrelevant_passages_stop_the_run_before_any_draft(self):
        class Anna(TW.Ret):
            def retrieve(self, claim, top_k=3):
                return [EvidencePassage(evidence_id="a", doc_title="Anna University", source_text=PAGES["Anna University"], retrieval_score=0.33),
                        EvidencePassage(evidence_id="h", doc_title="Hardbag", source_text=PAGES["Hardbag"], retrieval_score=0.33)], VerificationCost(model_calls=0)
        called = []
        ev, res = self._run(Anna(), lambda q, ps: called.append(1) or ("x y z a b c d.", VerificationCost()))
        self.assertIsNone(res)
        self.assertEqual(ev[-1]["type"], "insufficient_evidence")
        self.assertIn("VIT University", ev[-1]["message"])
        self.assertEqual(called, [])  # the generator was never asked to write an answer from unrelated text
        self.assertFalse(any(e["type"] == "result" for e in ev))

    def test_search_hits_that_are_all_rejected_say_so_instead_of_answering(self):
        class Wiki(FakeWiki):
            def get(self, url, params=None, timeout=None):
                if params and "gsrsearch" in params:
                    pages = [{"title": t, "index": i + 1, "extract": PAGES[t], "fullurl": "https://en.wikipedia.org/wiki/" + t.replace(" ", "_")} for i, t in enumerate(["Anna University", "Balut (food)"])]
                    return SimpleNamespace(status_code=200, headers={}, raise_for_status=lambda: None, json=lambda: {"query": {"pages": pages}})
                return super().get(url, params, timeout)
        called = []
        ev, res = self._run(WikipediaRetriever(session=Wiki(), min_interval=0.0), lambda q, ps: called.append(1) or ("x", VerificationCost()), question="Who founded the Zorblax Quantum Institute and when?")
        self.assertIsNone(res)
        self.assertEqual(ev[-1]["type"], "insufficient_evidence")
        self.assertIn("none of them is about Zorblax Quantum Institute", ev[-1]["message"])
        self.assertEqual(called, [])

    def test_generator_saying_insufficient_context_stops_the_run(self):
        good = TW.Ret()
        ev, res = self._run(good, lambda q, ps: (INSUFFICIENT, VerificationCost(model_calls=1)), question="Who wrote Frankenstein?")
        self.assertIsNone(res)
        self.assertEqual(ev[-1]["type"], "insufficient_evidence")
        self.assertEqual(ev[-1]["stage"], "draft")

    def test_fragment_draft_is_replaced_by_a_labelled_extractive_draft(self):
        ev, res = self._run(TW.Ret(), lambda q, ps: ("Mary Shelley 1818", VerificationCost(model_calls=1)), question="Who wrote Frankenstein?")
        dr = next(e for e in ev if e["type"] == "stage" and e["stage"] == "draft" and e["status"] == "done")
        self.assertIn("extractive fallback", dr["data"]["draft_source"])
        self.assertNotEqual(dr["data"]["draft"], "Mary Shelley 1818")
        passage_text = " ".join(p.source_text for p in TW.PASSAGES)
        for sent in dr["data"]["draft"].split(". "):
            self.assertIn(sent.strip().rstrip("."), passage_text)  # every sentence is verbatim from a retrieved passage

    def test_off_topic_claims_are_reported_not_verified_or_cited(self):
        draft = "Mary Shelley wrote Frankenstein. Frankenstein was first published in 1818. Hardbag is a subgenre of hard house music. Anna University is located in Chennai."
        ev, res = self._run(TW.Ret(), lambda q, ps: (draft, VerificationCost(model_calls=1)), question="Who wrote Frankenstein and when was it first published?")
        self.assertIsNotNone(res)
        off = {o["text"] for o in res["off_topic_claims"]}
        self.assertEqual(off, {"Hardbag is a subgenre of hard house music.", "Anna University is located in Chennai."})
        in_ledger = {l["text"] for l in res["ledger"]}
        self.assertTrue(in_ledger.isdisjoint(off))
        self.assertNotIn("Hardbag", res["answer"])
        self.assertEqual(res["metrics"]["claims_off_topic"], 2)
        verified_texts = {e["claim_text"] for e in ev if e["type"] == "verify" and e["phase"] == "done"}
        self.assertTrue(verified_texts.isdisjoint(off))

    def test_claim_filter_is_strict_for_factoid_and_permissive_for_explanatory_questions(self):
        who = analyse_question("Who created VIT University and when?")
        self.assertTrue(claim_is_on_topic(who, "VIT was founded in 1984 by G. Viswanathan."))
        self.assertFalse(claim_is_on_topic(who, "VIT organized an athletic event in March 2024."))  # names the entity but is not about its creation
        self.assertTrue(claim_is_on_topic(who, "Anything at all.", is_first=True))
        why = analyse_question("What caused the fall of the Western Roman Empire?")
        self.assertTrue(claim_is_on_topic(why, "In 376 the Goths entered the Roman Empire."))  # explanation that names the entity
        self.assertFalse(claim_is_on_topic(why, "Edward Gibbon was a historian."))

    def test_passage_gate_accepts_entity_passages_and_rejects_unrelated_ones(self):
        a = analyse_question("Who created VIT University and when?")
        mk = lambda t: EvidencePassage(evidence_id=t, doc_title=t, source_text=PAGES[t])  # noqa: E731
        self.assertTrue(passages_support_question(a, [mk("Vellore Institute of Technology"), mk("Hardbag")])["ok"])
        bad = passages_support_question(a, [mk("Anna University"), mk("Balut (food)")])
        self.assertFalse(bad["ok"])
        self.assertIn("VIT University", bad["reason"])


class TestGeneration(unittest.TestCase):
    def test_prompt_asks_for_a_direct_grounded_answer_and_offers_a_way_out(self):
        p = build_prompt("Who created VIT University and when?", [EvidencePassage(evidence_id="a", doc_title="Vellore Institute of Technology", source_text="Founded in 1984.")])
        for needle in ("Who created VIT University and when?", "[1] Vellore Institute of Technology: Founded in 1984.", "directly answers the question", "ONLY", INSUFFICIENT,
                       "Do not mention other topics", "No numbering"):
            self.assertIn(needle, p)

    def test_list_markers_never_become_claims(self):
        self.assertEqual(clean_draft("1. Mary wrote it.\n2. It appeared in 1818.\n- A third point."), "Mary wrote it. It appeared in 1818. A third point.")
        self.assertTrue(is_insufficient("INSUFFICIENT_CONTEXT"))
        self.assertFalse(is_well_formed("G. Viswanathan 1984 VIT"))
        self.assertTrue(is_well_formed("G. Viswanathan founded the institute in 1984. It became a university in 2001."))

    def test_one_retry_on_a_fragment_and_the_cost_covers_both_calls(self):
        replies = ["Mary Shelley 1818", "Mary Shelley wrote the novel Frankenstein. It was published in 1818."]
        seen = []

        class Client:
            class models:
                @staticmethod
                def list():
                    return SimpleNamespace(data=[SimpleNamespace(id="m")])

            class chat:
                class completions:
                    @staticmethod
                    def create(model, messages, temperature, max_tokens):
                        seen.append(messages[0]["content"])
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=replies[len(seen) - 1]))], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        text, cost = generate_draft("Who wrote Frankenstein?", TW.PASSAGES, client=Client)
        self.assertEqual(text, replies[1])
        self.assertEqual((len(seen), cost.model_calls, cost.total_tokens), (2, 2, 30))
        self.assertIn("not written as complete sentences", seen[1])

    def test_extractive_fallback_uses_only_verbatim_relevant_sentences(self):
        a = analyse_question("Who created VIT University and when?")
        ps = [EvidencePassage(evidence_id="v", doc_title="Vellore Institute of Technology", source_text=PAGES["Vellore Institute of Technology"] + " The university runs a festival every February in the main campus.")]
        d = extractive_draft(a, ps)
        self.assertIn("founded in 1984", d)
        self.assertNotIn("festival", d)


class TestCitationsFollowSupport(unittest.TestCase):
    def test_claim_is_cited_to_the_passage_that_supports_it(self):
        ps = [EvidencePassage(evidence_id="1", doc_title="Mary Shelley", source_text="Mary Shelley was born in London in 1797.", metadata={"url": "https://x/ms"}),
              EvidencePassage(evidence_id="2", doc_title="Frankenstein", source_text="Frankenstein is an 1818 Gothic novel written by Mary Shelley.", metadata={"url": "https://x/f"})]

        class R2(TW.Ret):
            def retrieve(self, claim, top_k=3):
                return list(ps), VerificationCost(model_calls=0)
        b = TW.backend(retriever_factory=lambda s: R2(), draft_fn=lambda q, p: ("Frankenstein is an 1818 Gothic novel written by Mary Shelley. Mary Shelley wrote Frankenstein.", VerificationCost(model_calls=1)))
        res = W.run_pipeline(b, W.RunConfig(question="Who wrote Frankenstein?", budget=4, policy="StaticFIFO"), lambda e: None)
        self.assertTrue(res["answer_items"])
        for it in res["answer_items"]:
            self.assertEqual((it["title"], it["url"]), ("Frankenstein", "https://x/f"))


if __name__ == "__main__":
    unittest.main()
