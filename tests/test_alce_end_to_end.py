"""Tests for scripts/run_alce_end_to_end.py (metrics and one controller run with scripted components; no LLM)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_alce_end_to_end as E  # noqa: E402
from src.claims.models import ClaimRecord, ClaimType, DecompositionStatus  # noqa: E402
from src.controller.interfaces import MockRepairer, MockRetriever  # noqa: E402
from src.controller.models import RepairResult  # noqa: E402
from src.controller.policy import ControllerPolicyConfig, StaticAllocationPolicy  # noqa: E402
from src.evidence.assessor import MockDeterministicAssessor  # noqa: E402
from src.evidence.models import AssessmentLabel, EvidenceAssessment, VerificationCost  # noqa: E402


class TestAlceMetric(unittest.TestCase):
    def test_normalisation_and_presence(self):
        self.assertEqual(E.normalize_answer("The Beatles!"), "beatles")
        self.assertTrue(E.exact_presence(["the Beatles"], "It was recorded by beatles in 1965."))
        self.assertFalse(E.exact_presence(["Rolling Stones"], "It was recorded by the Beatles."))
        self.assertFalse(E.exact_presence([""], "anything"))  # an empty short answer never matches

    def test_str_em_is_mean_over_disambiguated_questions(self):
        qa = [{"short_answers": ["Fred"]}, {"short_answers": ["Ron", "Ronald"]}, {"short_answers": ["Percy"]}]
        self.assertAlmostEqual(E.str_em(qa, "Fred and Ronald played the brothers."), 2 / 3)
        self.assertEqual(E.str_em([], "text"), 0.0)

    def test_citations_are_stripped_so_a_title_cannot_fake_a_match(self):
        answer = "He played a role [Rupert Grint]."
        self.assertEqual(E.strip_citations(answer), "He played a role .")
        self.assertFalse(E.exact_presence(["Rupert Grint"], E.strip_citations(answer)))
        self.assertTrue(E.exact_presence(["Rupert Grint"], answer))  # unstripped text WOULD have matched


class _Assessor(MockDeterministicAssessor):
    """Contradicts any claim mentioning Glasgow; supports everything else (including the repaired text)."""

    def assess(self, claim, passages):
        bad = "Glasgow" in claim.normalized_text
        return EvidenceAssessment(claim_id=claim.claim_id, evidence_ids=[p.evidence_id for p in passages], supporting_spans={},
                                  label=AssessmentLabel.CONTRADICTED if bad else AssessmentLabel.SUPPORTED, confidence=0.9,
                                  rationale="scripted", cost=VerificationCost(model_calls=1, latency_seconds=0.1))


class _Repairer(MockRepairer):
    def repair(self, claim, assessment, passages):
        rev = "James and Oliver Phelps were born in the Sutton Coldfield area of Birmingham, England on 25 February 1986."
        return RepairResult(claim_id=claim.claim_id, original_text=claim.normalized_text, proposed_revision=rev, repair_reason="scripted",
                            supporting_evidence_ids=["p1"], confidence=0.9, cost=VerificationCost(model_calls=1)), VerificationCost(model_calls=1)


def _prep():
    text = ("James and Oliver Phelps played Fred and George Weasley in the films. "
            "James and Oliver Phelps were born in the Sutton Coldfield area of Birmingham, England on 25 February 1986.")
    claims = []
    for i, t in enumerate(["James and Oliver Phelps played Fred and George Weasley.", "They were born in Glasgow, Scotland in 1975."]):
        claims.append(E.claim_to_dict(ClaimRecord(claim_id=f"c{i}", source_sentence_id=f"s{i}", original_text=t, normalized_text=t,
                                                  claim_type=ClaimType.FACTUAL_RELATION, decomposition_status=DecompositionStatus.ATOMIC,
                                                  triage_score=0.9 - 0.3 * i)))
    return {"sample_id": "q1", "question": "Who played the twins?", "qa_pairs": [{"short_answers": ["Sutton Coldfield"]}, {"short_answers": ["Fred"]}],
            "passages": [{"evidence_id": "p1", "doc_title": "James and Oliver Phelps", "source_text": text, "retrieval_score": 0.9}],
            "draft": "The twins played Fred and George. They were born in Glasgow.", "claims": claims, "prep_cost": {}}


class TestRunOne(unittest.TestCase):
    def _run(self, budget=8):
        return E.run_one(_prep(), StaticAllocationPolicy(ControllerPolicyConfig(enable_repair=True)), budget, MockRetriever(), _Assessor(), _Repairer(), "t")

    def test_verify_repair_reverify_flow_and_repair_gain(self):
        r = self._run()
        self.assertEqual(r["n_claims"], 2)
        self.assertEqual(r["n_flagged"], 1)               # the Glasgow claim was contradicted
        self.assertEqual((r["n_repair"], r["n_reverified_pass"]), (1, 1))
        self.assertEqual(r["str_em_final"], 1.0)          # repaired text carries 'Sutton Coldfield'; first claim carries 'Fred'
        self.assertEqual(r["str_em_final_no_repair"], 0.5)  # without the repaired claim only 'Fred' is in the answer
        self.assertNotIn("Glasgow", r["final_answer"])
        self.assertEqual(r["model_calls"], 4)             # 2 verifications + 1 repair + 1 re-verification

    def test_tight_budget_limits_what_reaches_the_answer(self):
        r = self._run(budget=1)
        self.assertLessEqual(r["model_calls"], 1)
        self.assertEqual(r["n_checked"], 1)
        self.assertEqual(r["n_repair"], 0)
        self.assertLess(r["str_em_final"], 1.0)

    def test_ungrounded_proxy_counts(self):
        r = self._run()
        self.assertEqual(r["ungrounded_total"], 1)        # the Glasgow/1975 claim has no supporting passage
        self.assertEqual(r["ungrounded_flagged"], 1)      # and the (scripted) verifier flagged it
        self.assertEqual(r["grounded_flagged"], 0)
        self.assertEqual(r["ungrounded_in_final"], 0)

    def test_question_baselines(self):
        b = E.question_baselines(_prep())
        self.assertEqual(b["retrieval_str_em"], 1.0)      # both short answers occur in the retrieved passage
        self.assertEqual(b["draft_str_em"], 0.5)          # the draft mentions Fred but not Sutton Coldfield


class TestAggregation(unittest.TestCase):
    def test_boot_ci_and_aggregate_average_over_seeds(self):
        rows = [[{"sample_id": "a", "str_em_final": 1.0}, {"sample_id": "b", "str_em_final": 0.0}],
                [{"sample_id": "a", "str_em_final": 0.0}, {"sample_id": "b", "str_em_final": 0.0}]]
        agg = E.aggregate({(2, "k"): rows}, [], ["a", "b"])
        self.assertTrue(np.allclose(agg[(2, "k")], [0.5, 0.0]))
        self.assertIsNone(E.boot_ci(np.array([1.0, 0.0])))  # too few questions for an interval
        ci = E.boot_ci(np.linspace(0, 1, 30))
        self.assertTrue(ci[0] < 0.5 < ci[1])


SMOKE = ROOT / "experiments" / "runs" / "alce_e2e_20261029_015253_smoke.json"


@unittest.skipUnless(SMOKE.exists(), "end-to-end smoke artifact not present")
class TestSummarizer(unittest.TestCase):
    def test_renders_loss_decomposition_budget_tables_and_proxy_section(self):
        import json
        import summarize_alce_end_to_end as S
        text = S.render(json.loads(SMOKE.read_text(encoding="utf-8")))
        for needle in ("Where the answer-bearing content is lost", "retrieval loss", "generation loss", "## Budget 2", "## Budget 4",
                       "Paired differences in final STR-EM", "PROXY label (not gold)", "LinUCB", "LexicalSupport", "NOT binding"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
