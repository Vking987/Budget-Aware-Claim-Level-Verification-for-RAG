"""Regression tests for claim-faithful final-answer synthesis, driven by the saved live Weasley trace
(experiments/runs/alce_controlled_repair_demo_v2_trace.json).

In that trace the contradicted "born ... 1996" claim ended UNRESOLVED after repair and failed re-verification, yet the
1B writer's answer repeated unverified birth-date sentences, school facts nobody verified, and duplicated paragraphs.
Here the trace's real claims, evidence and repair step are rebuilt and that real answer is fed back as the LLM output.
"""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus
from src.controller.models import ControllerAction, ControllerTrace, StoppingReason, TraceStep
from src.evidence.models import EvidencePassage
from src.synthesis.citations import check_citations
from src.synthesis.grounding import build_items, claim_tokens, stitch_answer, tokens, validate_answer
from src.synthesis.synthesizer import LLMAnswerSynthesizer

TRACE = Path(__file__).resolve().parent.parent / "experiments" / "runs" / "alce_controlled_repair_demo_v2_trace.json"

STITCHED = (
    "James and Oliver Phelps played Fred and George Weasley and are identical twin British actors [James and Oliver Phelps].\n"
    "Rupert Grint played Ron Weasley [Ron Weasley].\n"
    "Chris Rankin played Percy Weasley [Chris Rankin]."
)
NOTE = "Some claims could not be verified and are not included."


@unittest.skipUnless(TRACE.exists(), "supplied Weasley trace artifact not present")
class TestWeaselTraceGrounding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(TRACE.read_text(encoding="utf-8"))

    # ---- rebuild the run's inputs from the trace -------------------------------------------------
    def _inputs(self, reverification_verified=None):
        d = copy.deepcopy(self.data)
        passages = [EvidencePassage(evidence_id=p["evidence_id"], doc_title=p["doc_title"], source_text=p["text"],
                                    retrieval_score=p["retrieval_score"]) for p in d["evidence_passages"]]
        claims = [ClaimRecord(claim_id=c["claim_id"], source_sentence_id=c["claim_id"], original_text=c["text"],
                              normalized_text=c["text"], claim_type=ClaimType.FACTUAL_RELATION,
                              decomposition_status=DecompositionStatus.ATOMIC, state=ClaimState(c["state"]),
                              triage_score=c["triage_score"]) for c in d["claims"]]
        steps = []
        for s in d["trace"]["steps"]:
            if s["action"] == "STOP":
                continue
            if s["action"] == "REPAIR" and reverification_verified is not None:
                s["repair_data"]["reverification"]["verified"] = reverification_verified
            steps.append(TraceStep(step_number=s["step_number"], claim_id=s["claim_id"], claim_text=s["claim_text"],
                                   triage_score=s["triage_score"], previous_state=s["previous_state"], new_state=s["new_state"],
                                   action=ControllerAction(s["action"]), selection_rationale=s["selection_rationale"],
                                   assessment_label=s["assessment_label"], repair_data=s["repair_data"]))
        trace = ControllerTrace(run_id="weasley", policy_name="ContextualMAB_LinUCB", policy_config={}, initial_budget={},
                                final_budget_remaining={}, total_cost={}, total_steps=len(steps),
                                stopping_reason=StoppingReason.NO_ELIGIBLE_ACTIONS, claims_summary={}, steps=steps)
        store = {c.claim_id: list(passages) for c in claims}
        return claims, store, trace, d

    def _synthesize(self, llm_text, reverification_verified=None, raise_error=False):
        claims, store, trace, d = self._inputs(reverification_verified)
        client = MagicMock()
        if raise_error:
            client.chat.completions.create.side_effect = RuntimeError("server down")
        else:
            resp = MagicMock()
            resp.choices[0].message.content = llm_text
            resp.choices[0].finish_reason = "stop"
            resp.usage.prompt_tokens, resp.usage.completion_tokens = 100, 50
            client.chat.completions.create.return_value = resp
        res = LLMAnswerSynthesizer(client=client).synthesize(
            question=d["question"], initial_draft_answer=d["initial_draft"], claims=claims, evidence_store=store,
            assessments={}, trace=trace)
        return res, claims, store, d, client

    @staticmethod
    def _sentences(answer):
        import re
        return [s.strip() for ln in answer.splitlines() for s in re.split(r"(?<=[.!?])\s+", ln) if s.strip()]

    # ---- the saved trace really is the bug scenario ---------------------------------------------
    def test_fixture_is_the_bug_scenario(self):
        states = {c["claim_id"]: c["state"] for c in self.data["claims"]}
        self.assertEqual(sorted(k for k, v in states.items() if v == "VERIFIED"), ["s_0_c0", "s_1_c0", "s_3_c0", "s_4_c0"])
        self.assertEqual(states["s_2_c0"], "UNRESOLVED")
        rep = [s for s in self.data["trace"]["steps"] if s["action"] == "REPAIR"][0]
        self.assertFalse(rep["repair_data"]["reverification"]["verified"])
        self.assertIn("1986", rep["repair_data"]["proposed_revision"])  # failed repair text
        self.assertIn("1996", [c["text"] for c in self.data["claims"] if c["claim_id"] == "s_2_c0"][0])
        self.assertIn("1986", self.data["final_answer"])  # the buggy answer leaked the unverified repair content

    # ---- main regression: the real bad LLM answer is rejected and replaced ----------------------
    def test_real_bad_answer_is_rejected_and_replaced_by_stitched_answer(self):
        res, claims, store, d, client = self._synthesize(self.data["final_answer"])
        self.assertEqual(client.chat.completions.create.call_count, 1)  # the optional LLM step did run
        self.assertEqual(res.polish_status, "llm_rejected_stitched")
        self.assertEqual(res.synthesis_method, "deterministic_stitched")
        self.assertEqual(res.answer_violations, [])
        self.assertEqual(res.final_answer, STITCHED + "\n" + NOTE)
        self.assertTrue(any("LLM rewrite rejected" in n for n in res.uncertainty_notes))

    def test_incorrect_and_unverified_content_never_appears(self):
        res, *_ = self._synthesize(self.data["final_answer"])
        for banned in ("1996", "1986", "Sutton Coldfield", "Northgate", "Dereham", "Arthur Terry", "Little Sutton", "educated", "attended"):
            self.assertNotIn(banned, res.final_answer, banned)
        # the unresolved claim is disclosed only as a fixed, claim-free sentence and kept in the structured fields
        self.assertIn(NOTE, res.final_answer)
        self.assertTrue(any("1996" in t for t in res.unresolved_claims + res.contradicted_claims))
        self.assertFalse(any("1996" in t for t in res.verified_claims))

    def test_all_four_verified_claims_are_preserved(self):
        res, claims, *_ = self._synthesize(self.data["final_answer"])
        answer_tokens = claim_tokens(res.final_answer)
        verified = [c for c in claims if c.state == ClaimState.VERIFIED]
        self.assertEqual(len(verified), 4)
        for c in verified:
            self.assertLessEqual(claim_tokens(c.normalized_text), answer_tokens, c.normalized_text)

    def test_duplicates_are_removed_and_facts_combined(self):
        res, *_ = self._synthesize(self.data["final_answer"])
        sents = [" ".join(tokens(s)) for s in self._sentences(res.final_answer)]
        self.assertEqual(len(sents), len(set(sents)), "duplicate sentences in the final answer")
        self.assertEqual(len(self._sentences(res.final_answer)), 4)  # 3 fact sentences (two claims merged) + fixed note
        self.assertEqual(res.final_answer.count("Rupert Grint played Ron Weasley"), 1)
        self.assertIn("played Fred and George Weasley and are identical twin British actors", res.final_answer)

    def test_citations_resolve_and_support_their_claims(self):
        res, claims, store, d, _ = self._synthesize(self.data["final_answer"])
        passages = [{"evidence_id": p["evidence_id"], "doc_title": p["doc_title"]} for p in d["evidence_passages"]]
        chk = check_citations(res.final_answer, passages)
        self.assertEqual(chk["unresolved"], [])
        self.assertEqual(sorted(chk["resolved"]), ["Chris Rankin", "James and Oliver Phelps", "Ron Weasley"])
        self.assertEqual(res.cited_sources, ["Chris Rankin", "James and Oliver Phelps", "Ron Weasley"])  # not all 5 retrieved
        by_id = {p["evidence_id"]: p["text"] for p in d["evidence_passages"]}
        expected = {"s_0_c0": "qdrant_300", "s_1_c0": "qdrant_300", "s_3_c0": "qdrant_325", "s_4_c0": "qdrant_309"}
        got = {cc["claim_id"]: cc["evidence_ids"] for cc in res.claim_citations}
        self.assertEqual({k: v[0] for k, v in got.items()}, expected)
        self.assertNotIn("s_2_c0", got)  # the unresolved claim has no citation entry
        for cc in res.claim_citations:  # every cited passage really contains most of the claim's content words
            ctoks = claim_tokens(cc["claim"])
            ptoks = claim_tokens(by_id[cc["evidence_ids"][0]])
            self.assertGreaterEqual(len(ctoks & ptoks) / len(ctoks), 0.5, cc["claim"])
        # each answer line cites only sources that support the claims on that line
        for line in res.final_answer.splitlines()[:3]:
            titles = [m for m in chk["resolved"] if f"[{m}]" in line]
            self.assertEqual(len(titles), 1, line)

    # ---- other ways the writer can go wrong ------------------------------------------------------
    def test_llm_answer_that_restates_the_1996_claim_is_rejected(self):
        bad = STITCHED + "\nJames and Oliver Phelps were born in the Sutton Coldfield area of Birmingham, England on 25 February 1996 [James and Oliver Phelps]."
        res, *_ = self._synthesize(bad)
        self.assertEqual(res.polish_status, "llm_rejected_stitched")
        self.assertNotIn("1996", res.final_answer)

    def test_clean_llm_rewrite_is_accepted_unchanged(self):
        res, *_ = self._synthesize(STITCHED)
        self.assertEqual(res.polish_status, "llm_accepted")
        self.assertEqual(res.synthesis_method, "llm_grounded")
        self.assertEqual(res.final_answer, STITCHED + "\n" + NOTE)
        self.assertEqual(res.answer_violations, [])

    def test_llm_that_drops_a_verified_claim_is_rejected(self):
        res, *_ = self._synthesize("Rupert Grint played Ron Weasley [Ron Weasley].")
        self.assertEqual(res.polish_status, "llm_rejected_stitched")
        self.assertIn("Chris Rankin played Percy Weasley", res.final_answer)

    def test_llm_with_wrong_citation_is_rejected(self):
        res, *_ = self._synthesize(STITCHED.replace("[Chris Rankin]", "[Mark Williams (actor)]"))
        self.assertEqual(res.polish_status, "llm_rejected_stitched")
        self.assertIn("[Chris Rankin]", res.final_answer)

    def test_llm_recombining_verified_words_into_a_new_claim_is_rejected(self):
        res, *_ = self._synthesize("Rupert Grint played Percy Weasley [Ron Weasley]. Chris Rankin played Ron Weasley [Chris Rankin].")
        self.assertEqual(res.polish_status, "llm_rejected_stitched")
        self.assertNotIn("Rupert Grint played Percy", res.final_answer)

    def test_llm_failure_falls_back_to_stitched(self):
        res, *_ = self._synthesize("", raise_error=True)
        self.assertEqual(res.synthesis_method, "fallback_grounded")
        self.assertEqual(res.final_answer, STITCHED + "\n" + NOTE)
        self.assertIsNotNone(res.error)

    # ---- a repaired claim counts only if re-verification passed ---------------------------------
    def test_repaired_claim_is_included_only_when_reverification_passes(self):
        failed, *_ = self._synthesize(self.data["final_answer"], reverification_verified=False)
        self.assertNotIn("1986", failed.final_answer)
        passed, *_ = self._synthesize(self.data["final_answer"], reverification_verified=True)
        self.assertIn("born in the Sutton Coldfield area of Birmingham, England on 25 February 1986", passed.final_answer)
        self.assertNotIn("1996", passed.final_answer)
        self.assertIn("James and Oliver Phelps", passed.cited_sources)  # 1986 literally occurs in the cited passage

    def test_abstention_never_quotes_unresolved_claims(self):
        claims, store, trace, d = self._inputs()
        for c in claims:
            c.state = ClaimState.UNRESOLVED
        client = MagicMock()
        res = LLMAnswerSynthesizer(client=client).synthesize(
            question=d["question"], initial_draft_answer=d["initial_draft"], claims=claims, evidence_store=store,
            assessments={}, trace=None)
        client.chat.completions.create.assert_not_called()
        self.assertEqual(res.synthesis_method, "safe_abstention")
        for banned in ("1996", "Phelps", "Rankin", "Grint"):
            self.assertNotIn(banned, res.final_answer)
        self.assertIn("insufficient", res.final_answer.lower())

    # ---- validator unit checks on the real items -----------------------------------------------
    def test_validator_flags_each_failure_mode(self):
        claims, store, trace, d = self._inputs()
        items = build_items(claims, {}, store, {c.claim_id for c in claims if c.state == ClaimState.VERIFIED})
        excluded = [c.normalized_text for c in claims if c.state != ClaimState.VERIFIED]
        self.assertEqual(validate_answer(stitch_answer(items), items, excluded), [])
        dup = STITCHED + "\nRupert Grint played Ron Weasley [Ron Weasley]."
        self.assertTrue(any("duplicate" in v for v in validate_answer(dup, items, excluded)))
        self.assertTrue(any("dropped" in v for v in validate_answer("Rupert Grint played Ron Weasley [Ron Weasley].", items, excluded)))
        self.assertTrue(any("1996" in v for v in validate_answer(STITCHED + "\nThe twins were born in 1996.", items, excluded)))
        self.assertTrue(any("missing citation" in v for v in validate_answer(STITCHED.replace(" [Ron Weasley]", ""), items, excluded)))
        self.assertTrue(any("does not support" in v for v in validate_answer(STITCHED.replace("[Ron Weasley]", "[Chris Rankin]"), items, excluded)))


if __name__ == "__main__":
    unittest.main()
