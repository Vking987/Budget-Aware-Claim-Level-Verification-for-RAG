"""Tests for scripts/replay_answer_handling.py (scripted verifier/repairer; no model, no network)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import replay_answer_handling as P  # noqa: E402
import run_alce_end_to_end as E  # noqa: E402
from src.claims.models import ClaimRecord, ClaimType, DecompositionStatus  # noqa: E402
from src.controller.interfaces import MockRepairer, MockRetriever  # noqa: E402
from src.controller.models import RepairResult  # noqa: E402
from src.controller.policy import ControllerPolicyConfig, StaticAllocationPolicy  # noqa: E402
from src.evidence.assessor import MockDeterministicAssessor  # noqa: E402
from src.evidence.models import AssessmentLabel, EvidenceAssessment, VerificationCost  # noqa: E402

TEXTS = ["Fred was played by James Phelps.", "They were born in Glasgow.", "The twins maybe liked football.", "Oliver Phelps played George."]


class _Assessor(MockDeterministicAssessor):
    def assess(self, claim, passages):
        t = claim.normalized_text
        lab = AssessmentLabel.CONTRADICTED if "Glasgow" in t else AssessmentLabel.INSUFFICIENT if "maybe" in t else AssessmentLabel.SUPPORTED
        return EvidenceAssessment(claim_id=claim.claim_id, evidence_ids=[p.evidence_id for p in passages], supporting_spans={}, label=lab,
                                  confidence=0.9, rationale="s", cost=VerificationCost(model_calls=1))


class _Repairer(MockRepairer):
    def __init__(self, revision):
        self.revision = revision

    def repair(self, claim, assessment, passages):
        return RepairResult(claim_id=claim.claim_id, original_text=claim.normalized_text, proposed_revision=self.revision, repair_reason="s",
                            supporting_evidence_ids=["p1"], confidence=0.9, cost=VerificationCost(model_calls=1)), VerificationCost(model_calls=1)


def _prep():
    claims = [E.claim_to_dict(ClaimRecord(claim_id=f"c{i}", source_sentence_id=f"s{i}", original_text=t, normalized_text=t, claim_type=ClaimType.FACTUAL_RELATION,
                                          decomposition_status=DecompositionStatus.ATOMIC, triage_score=0.5)) for i, t in enumerate(TEXTS)]
    return {"sample_id": "q", "question": "Q?", "qa_pairs": [{"short_answers": ["Birmingham"]}, {"short_answers": ["football"]}, {"short_answers": ["George"]}],
            "passages": [{"evidence_id": "p1", "doc_title": "T", "source_text": "The Phelps twins were born in Birmingham.", "retrieval_score": 0.9}],
            "draft": " ".join(TEXTS), "claims": claims, "prep_cost": {}}


class TestStrategies(unittest.TestCase):
    def setUp(self):
        orig = E.VerificationController
        E.VerificationController = P._Capture
        self.addCleanup(lambda: setattr(E, "VerificationController", orig))

    def _row(self, revision):
        row = E.run_one(_prep(), StaticAllocationPolicy(ControllerPolicyConfig(enable_repair=True)), 10, MockRetriever(), _Assessor(), _Repairer(revision), "t")
        claims, trace, store = P._Capture.last
        out = P.claim_outcomes(claims, trace)
        prep = _prep()
        return row, {s: P.assemble(s, claims, out, store, prep["qa_pairs"]) for s in P.STRATEGIES}, out

    def test_failed_repair_conservative_keeps_insufficient_and_excludes_contradicted(self):
        row, st, out = self._row("They were born in Glasgow, Scotland.")  # revision still contradicted -> re-verification fails
        self.assertEqual(out["repairs_pass"], {})
        self.assertAlmostEqual(st["S0_current"]["str_em"], row["str_em_final"])
        self.assertNotIn("football", st["S0_current"]["answer"])           # INSUFFICIENT dropped by the current rule
        for s in ("S1_conservative", "S2_repair_first"):
            self.assertIn("football", st[s]["answer"])                      # retained
            self.assertNotIn("Glasgow", st[s]["answer"])                    # CONTRADICTED excluded
        self.assertEqual(st["S1_conservative"]["answer"], st["S2_repair_first"]["answer"])  # no passed repair -> identical fallback
        self.assertEqual(st["S1_conservative"]["n_insufficient_kept"], 1)

    def test_passed_repair_is_used_by_s0_and_s2_but_not_s1(self):
        row, st, out = self._row("They were born in Birmingham.")
        self.assertEqual(list(out["repairs_pass"]), ["c1"])
        self.assertIn("Birmingham", st["S0_current"]["answer"])
        self.assertIn("Birmingham", st["S2_repair_first"]["answer"])
        self.assertNotIn("Birmingham", st["S1_conservative"]["answer"])
        self.assertEqual(st["S2_repair_first"]["n_repairs_used"], 1)
        self.assertEqual(st["S1_conservative"]["n_repairs_used"], 0)

    def test_unverified_claims_are_dropped_in_every_strategy(self):
        E.run_one(_prep(), StaticAllocationPolicy(ControllerPolicyConfig(enable_repair=True)), 1, MockRetriever(), _Assessor(), _Repairer("x"), "t")
        claims, trace, store = P._Capture.last
        out = P.claim_outcomes(claims, trace)
        self.assertEqual(len(out["last"]), 1)
        for s in P.STRATEGIES:
            r = P.assemble(s, claims, out, store, _prep()["qa_pairs"])
            self.assertLessEqual(r["n_in_answer"], 1)


class TestGuards(unittest.TestCase):
    def test_analyse_runs_on_a_tiny_synthetic_run(self):
        import json
        import tempfile
        rows = [{"sample_id": "a", "n_claims": 4, "latency_seconds": 1.0, "strategies": {s: {"str_em": 0.1 * i, "n_in_answer": i, "n_insufficient_kept": 0,
                 "n_repairs_used": 0, "ungrounded_in_answer": 0} for i, s in enumerate(P.STRATEGIES)},
                 "strategy_counts": {"n_claims": 4, "n_verified": 2, "last_labels": {"SUPPORTED": 2, "INSUFFICIENT": 1}, "n_repairs": 1, "n_repairs_passed": 0}}] * 12
        d = {"manifest": {"run_id": "r", "llm_cache": {"hits": 1, "misses": 0}, "questions": "x", "seeds": [0], "budgets": [4]},
             "summary": {"draft_str_em": 0.3}, "per_question_runs": {"4|LinUCB": [rows]}}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.json"
            p.write_text(json.dumps(d))
            text = P.analyse(p, 4)
        for needle in ("S1_conservative", "S2_repair_first", "repair success", "LinUCB"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
