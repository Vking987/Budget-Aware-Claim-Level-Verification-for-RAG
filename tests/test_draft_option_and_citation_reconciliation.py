"""Regression tests: --draft (user-supplied draft, generation skipped), repair -> re-verification flow
through the full pipeline, and cited_sources reconciled with the claim -> passage map. No live services."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus  # noqa: E402
from src.controller.interfaces import MockRepairer, MockRetriever  # noqa: E402
from src.controller.models import ControllerAction, ControllerTrace, StoppingReason, TraceStep  # noqa: E402
from src.evidence.assessor import MockDeterministicAssessor  # noqa: E402
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost  # noqa: E402
from src.pipeline import RAGVerificationPipeline  # noqa: E402
from src.synthesis.synthesizer import LLMAnswerSynthesizer, MockAnswerSynthesizer  # noqa: E402

BORN = "James and Oliver Phelps were born in the Sutton Coldfield area of Birmingham, England on 25 February 1986."
PLAY = "James and Oliver Phelps are identical twin British actors who played Fred and George Weasley."
P_PHELPS = EvidencePassage(evidence_id="qdrant_300", doc_title="James and Oliver Phelps", source_text=PLAY + " " + BORN, retrieval_score=0.71)
P_WILLIAMS = EvidencePassage(evidence_id="qdrant_310", doc_title="Mark Williams (actor)",
                             source_text="Mark Williams plays Arthur Weasley in the Harry Potter films.", retrieval_score=0.66)
P_RON = EvidencePassage(evidence_id="qdrant_325", doc_title="Ron Weasley",
                        source_text="Rupert Grint plays Ron Weasley who has freckles.", retrieval_score=0.65)

DRAFT = ("James and Oliver Phelps played Fred and George Weasley. "
         "They were born in the Sutton Coldfield area of Birmingham, England. "
         "They were born in Glasgow, Scotland in 1975.")
REVISION = "James and Oliver Phelps were born in the Sutton Coldfield area of Birmingham, England."


def _cost():
    return VerificationCost(model_calls=1, prompt_tokens=100, completion_tokens=10, total_tokens=110, latency_seconds=0.5)


class _ScriptedAssessor(MockDeterministicAssessor):
    """Contradicts any text mentioning Glasgow; for the REVISED text, answers per `revision_label`."""

    def __init__(self, revision_label):
        self.revision_label = revision_label
        self.seen = []

    def assess(self, claim, passages):
        self.seen.append(claim.normalized_text)
        if "Glasgow" in claim.normalized_text:
            label = AssessmentLabel.CONTRADICTED
        elif claim.normalized_text == REVISION:
            label = self.revision_label
        else:
            label = AssessmentLabel.SUPPORTED
        return EvidenceAssessment(claim_id=claim.claim_id, evidence_ids=[p.evidence_id for p in passages], supporting_spans={},
                                  label=label, confidence=0.9, rationale="scripted", cost=_cost())


class _ScriptedRepairer(MockRepairer):
    def repair(self, claim, assessment, passages):
        from src.controller.models import RepairResult
        return RepairResult(claim_id=claim.claim_id, original_text=claim.normalized_text, proposed_revision=REVISION,
                            repair_reason="scripted", supporting_evidence_ids=["qdrant_300"], confidence=0.9, cost=_cost()), _cost()


def _pipeline(revision_label):
    retr = MockRetriever()
    retr.set_evidence_for_claim("init_query", [P_PHELPS, P_WILLIAMS, P_RON])
    assessor = _ScriptedAssessor(revision_label)
    pl = RAGVerificationPipeline(retriever=retr, assessor=assessor, repairer=_ScriptedRepairer(),
                                 synthesizer=MockAnswerSynthesizer())
    return pl, assessor


class TestDraftOption(unittest.TestCase):
    def test_supplied_draft_skips_generation_and_is_labelled(self):
        pl, _ = _pipeline(AssessmentLabel.SUPPORTED)
        with patch.object(pl, "_generate_draft", side_effect=AssertionError("generation must not run")):
            res = pl.run("Who played the Weasley twins?", draft_answer=DRAFT, policy_name="DeterministicCTS",
                         max_steps=12, max_model_calls=12)
        self.assertEqual(res.draft_source, "user_supplied")
        self.assertEqual(res.initial_draft, DRAFT)
        self.assertTrue(any("SKIPPED" in w for w in res.warnings))
        self.assertEqual(res.initial_cost["model_calls"], 0)
        self.assertEqual(len(res.claims), 3)  # rule-based decomposition ran on the supplied text
        self.assertTrue(all(c["triage_score"] is not None for c in res.claims))  # CTS ran
        self.assertEqual(res.to_dict()["draft_source"], "user_supplied")

    def test_default_path_still_generates_a_draft(self):
        pl, _ = _pipeline(AssessmentLabel.SUPPORTED)
        with patch.object(pl, "_generate_draft", return_value=("Generated draft sentence.", VerificationCost(model_calls=1))) as g:
            res = pl.run("q", policy_name="DeterministicCTS")
        g.assert_called_once()
        self.assertEqual(res.draft_source, "generated")
        self.assertEqual(res.initial_cost["model_calls"], 1)

    def test_blank_draft_is_rejected(self):
        pl, _ = _pipeline(AssessmentLabel.SUPPORTED)
        with self.assertRaises(ValueError):
            pl.run("q", draft_answer="   ")

    def test_budget_overrides_apply_and_default_is_unchanged(self):
        pl, _ = _pipeline(AssessmentLabel.SUPPORTED)
        res = pl.run("q", draft_answer=DRAFT, policy_name="DeterministicCTS")
        self.assertEqual((res.budget["max_steps"], res.budget["max_model_calls"]), (5, 8))
        pl2, _ = _pipeline(AssessmentLabel.SUPPORTED)
        res2 = pl2.run("q", draft_answer=DRAFT, policy_name="DeterministicCTS", max_steps=1, max_model_calls=1)
        self.assertEqual((res2.budget["max_steps"], res2.budget["max_model_calls"]), (1, 1))
        self.assertLessEqual(res2.steps_taken, 1)


class TestRepairReverificationThroughPipeline(unittest.TestCase):
    def _run(self, revision_label):
        pl, assessor = _pipeline(revision_label)
        res = pl.run("Who played the Weasley twins?", draft_answer=DRAFT, policy_name="DeterministicCTS",
                     max_steps=12, max_model_calls=12)
        repair_steps = [s for s in res.trace.steps if s.action == ControllerAction.REPAIR]
        return res, assessor, repair_steps

    def test_repair_runs_and_revision_is_reverified_before_it_is_cited(self):
        res, assessor, repair_steps = self._run(AssessmentLabel.SUPPORTED)
        self.assertEqual(len(repair_steps), 1)
        rv = repair_steps[0].repair_data["reverification"]
        self.assertTrue(rv["verified"])
        self.assertEqual(rv["cost"]["model_calls"], 1)
        self.assertIn(REVISION, assessor.seen)  # the assessor really saw the revised text
        # 3 initial verifications + 1 repair + 1 re-verification, all charged to the budget
        self.assertEqual(res.trace.total_cost["model_calls"], 5)
        self.assertIn(REVISION, res.final_answer)
        glasgow = [c for c in res.claims if "Glasgow" in c["text"]][0]
        self.assertEqual(glasgow["state"], "UNRESOLVED")  # repaired != verified
        self.assertNotIn("Glasgow", res.final_answer)

    def test_failed_reverification_keeps_revision_out_of_the_answer(self):
        res, assessor, repair_steps = self._run(AssessmentLabel.INSUFFICIENT)
        self.assertEqual(len(repair_steps), 1)
        self.assertFalse(repair_steps[0].repair_data["reverification"]["verified"])
        self.assertEqual(res.trace.total_cost["model_calls"], 5)  # re-verification still cost a call
        glasgow = [c for c in res.claims if "Glasgow" in c["text"]][0]
        self.assertEqual(glasgow["state"], "UNRESOLVED")
        syn = res.synthesis
        # neither the contradicted claim nor its un-reverified revision is presented as verified
        self.assertFalse(any("Glasgow" in t for t in syn["verified_claims"]))
        self.assertNotIn(REVISION, syn["verified_claims"])
        # it is disclosed as unverified/contradicted instead of being asserted
        self.assertTrue(any("Glasgow" in t for t in syn["unresolved_claims"] + syn["contradicted_claims"]))
        self.assertIn("remain unverified", res.final_answer)

    def test_every_step_has_cost_and_budget_snapshots(self):
        res, _, _ = self._run(AssessmentLabel.SUPPORTED)
        for s in res.trace.steps:
            if s.action == ControllerAction.STOP:
                continue
            self.assertIsNotNone(s.action_cost)
            self.assertIn("remaining_steps", s.remaining_budget)
            self.assertTrue(s.selection_rationale)


def _verified_claim(cid, text):
    return ClaimRecord(claim_id=cid, source_sentence_id="s", original_text=text, normalized_text=text,
                       claim_type=ClaimType.FACTUAL_RELATION, decomposition_status=DecompositionStatus.ATOMIC,
                       state=ClaimState.VERIFIED)


def _llm_synth():
    client = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = "James and Oliver Phelps were born in Sutton Coldfield, England. [James and Oliver Phelps]"
    resp.choices[0].finish_reason = "stop"
    resp.usage.prompt_tokens, resp.usage.completion_tokens = 50, 20
    client.chat.completions.create.return_value = resp
    return LLMAnswerSynthesizer(client=client)


class TestCitedSourcesReconciliation(unittest.TestCase):
    def _synth(self, claims, store, trace=None):
        return _llm_synth().synthesize(question="Who played the twins?", initial_draft_answer="draft", claims=claims,
                                       evidence_store=store, assessments={}, trace=trace)

    def test_cites_only_the_best_matching_passage_not_every_assessed_passage(self):
        claim = _verified_claim("c1", "They were born in the Sutton Coldfield area of Birmingham, England on 25 February 1986.")
        res = self._synth([claim], {"c1": [P_PHELPS, P_WILLIAMS, P_RON]})
        self.assertEqual(res.cited_sources, ["James and Oliver Phelps"])

    def test_never_cites_a_passage_that_does_not_support_the_claim_even_if_provenance_allows_it(self):
        claim = _verified_claim("c1", "They were born in the Sutton Coldfield area of Birmingham, England.")
        claim.evidence_provenance = {"supporting_spans": {"qdrant_310": ["Mark Williams"]}}  # provenance allows only P_WILLIAMS
        res = self._synth([claim], {"c1": [P_PHELPS, P_WILLIAMS, P_RON]})
        self.assertEqual(res.cited_sources, [])  # the Phelps passage supports it but is excluded; Williams does not support it
        self.assertTrue(any("no retrieved passage" in n for n in res.uncertainty_notes))

    def test_no_citation_when_no_passage_supports_the_claim(self):
        claim = _verified_claim("c1", "Quantum chromodynamics describes gluon interactions.")
        res = self._synth([claim], {"c1": [P_PHELPS, P_WILLIAMS]})
        self.assertEqual(res.cited_sources, [])  # previously every assessed passage was cited
        self.assertFalse(res.is_grounded)  # grounding needs at least one resolved supporting source

    def test_number_in_claim_must_literally_occur_in_the_cited_passage(self):
        wrong_year = _verified_claim("c1", "James and Oliver Phelps were born in the Sutton Coldfield area of Birmingham, England on 25 February 1996.")
        res = self._synth([wrong_year], {"c1": [P_PHELPS]})
        self.assertEqual(res.cited_sources, [])  # 1996 is not in the passage (it says 1986) -> no supporting citation

    def test_repaired_claim_is_cited_by_its_revision_text(self):
        claim = _verified_claim("c1", "Completely wrong original text about Glasgow.")
        claim.state = ClaimState.UNRESOLVED
        rep = {"proposed_revision": "Rupert Grint plays Ron Weasley who has freckles.", "supporting_evidence_ids": [],
               "reverification": {"label": "SUPPORTED", "confidence": 0.9, "verified": True}}
        trace = ControllerTrace(run_id="t", policy_name="p", policy_config={}, initial_budget={}, final_budget_remaining={},
                                total_cost={}, total_steps=1, stopping_reason=StoppingReason.ALL_CLAIMS_RESOLVED, claims_summary={},
                                steps=[TraceStep(step_number=1, claim_id="c1", claim_text="x", triage_score=0.5, previous_state="SELECTED",
                                                 new_state="UNRESOLVED", action=ControllerAction.REPAIR, selection_rationale="r",
                                                 assessment_label="CONTRADICTED", repair_data=rep)])
        res = self._synth([claim], {"c1": [P_PHELPS, P_WILLIAMS, P_RON]}, trace=trace)
        self.assertEqual(res.cited_sources, ["Ron Weasley"])

    def test_unverified_repair_contributes_no_citation(self):
        claim = _verified_claim("c1", "Completely wrong original text about Glasgow.")
        claim.state = ClaimState.UNRESOLVED
        rep = {"proposed_revision": "Rupert Grint plays Ron Weasley who has freckles.", "supporting_evidence_ids": [],
               "reverification": {"label": "INSUFFICIENT", "confidence": 0.4, "verified": False}}
        trace = ControllerTrace(run_id="t", policy_name="p", policy_config={}, initial_budget={}, final_budget_remaining={},
                                total_cost={}, total_steps=1, stopping_reason=StoppingReason.ALL_CLAIMS_RESOLVED, claims_summary={},
                                steps=[TraceStep(step_number=1, claim_id="c1", claim_text="x", triage_score=0.5, previous_state="SELECTED",
                                                 new_state="UNRESOLVED", action=ControllerAction.REPAIR, selection_rationale="r",
                                                 assessment_label="CONTRADICTED", repair_data=rep)])
        res = self._synth([claim], {"c1": [P_PHELPS, P_WILLIAMS, P_RON]}, trace=trace)
        self.assertEqual(res.cited_sources, [])


class TestAnswerAuditInPipeline(unittest.TestCase):
    def _run_with_answer(self, text):
        client = MagicMock()
        resp = MagicMock()
        resp.choices[0].message.content = text
        resp.choices[0].finish_reason = "stop"
        resp.usage.prompt_tokens, resp.usage.completion_tokens = 50, 20
        client.chat.completions.create.return_value = resp
        retr = MockRetriever()
        retr.set_evidence_for_claim("init_query", [P_PHELPS, P_WILLIAMS, P_RON])
        pl = RAGVerificationPipeline(retriever=retr, assessor=_ScriptedAssessor(AssessmentLabel.INSUFFICIENT),
                                     repairer=_ScriptedRepairer(), synthesizer=LLMAnswerSynthesizer(client=client))
        # only ONE supported claim; the birth claim is contradicted and its repair fails re-verification,
        # so no verified claim covers the birth topic (the live-demo situation)
        return pl.run("Who played the Weasley twins?",
                      draft_answer="James and Oliver Phelps played Fred and George Weasley. They were born in Glasgow, Scotland in 1975.",
                      policy_name="DeterministicCTS", max_steps=12, max_model_calls=12)

    def test_llm_answer_asserting_unverified_repair_content_is_rejected_and_replaced(self):
        # re-verification fails (INSUFFICIENT), yet the writer asserts the corrected birth date from the raw passage
        res = self._run_with_answer(
            "James and Oliver Phelps played Fred and George Weasley [James and Oliver Phelps]. "
            "James and Oliver Phelps were born in the Sutton Coldfield area of Birmingham, England on 25 February 1986 "
            "[James and Oliver Phelps].")
        self.assertEqual(res.synthesis["polish_status"], "llm_rejected_stitched")
        self.assertEqual(res.synthesis["synthesis_method"], "deterministic_stitched")
        self.assertNotIn("1986", res.final_answer)
        self.assertNotIn("1975", res.final_answer)
        self.assertNotIn("Glasgow", res.final_answer)
        self.assertIn("James and Oliver Phelps played Fred and George Weasley", res.final_answer)
        self.assertEqual(res.answer_validation["violations"], [])
        self.assertTrue(res.answer_validation["enforced"])
        self.assertTrue(any("failed grounding validation and was replaced" in w for w in res.warnings))
        self.assertTrue(all(r["backed"] for r in res.answer_audit))  # the lexical audit agrees
        self.assertEqual(res.to_dict()["answer_validation"], res.answer_validation)

    def test_valid_llm_rewrite_is_accepted(self):
        res = self._run_with_answer("James and Oliver Phelps played Fred and George Weasley [James and Oliver Phelps].")
        self.assertEqual(res.synthesis["polish_status"], "llm_accepted")
        self.assertEqual(res.synthesis["synthesis_method"], "llm_grounded")
        self.assertEqual(res.answer_validation["violations"], [])
        self.assertFalse(any("replaced by the" in w for w in res.warnings))


class TestDemoPlumbing(unittest.TestCase):
    def test_demo_draft_flag_marks_skip_and_saves_json(self):
        import demo
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "trace.json"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                demo.run_demo("Who played the twins?", dry_run=True, draft="James played Fred. Oliver played George.",
                              output_path=str(out), max_steps=6, max_calls=6)
            text = buf.getvalue()
            self.assertIn("SKIPPED", text)
            self.assertIn("DRAFT SOURCE: USER-SUPPLIED", text)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["draft_source"], "user_supplied")
            self.assertEqual(data["run_settings"]["budget"]["max_steps"], 6)
            self.assertIn("trace", data)


if __name__ == "__main__":
    unittest.main()
