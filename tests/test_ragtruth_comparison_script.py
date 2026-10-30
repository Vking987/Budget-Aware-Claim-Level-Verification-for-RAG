"""Plumbing/fairness tests for scripts/run_ragtruth_comparison.py (mock components, tiny fixture)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_ragtruth_comparison as R  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "ragtruth_sample.json"
CKPT = ROOT / "experiments" / "runs" / "linucb_warmstart_train100_300.json"


class TestComparisonScript(unittest.TestCase):
    def test_chunking_respects_limit_and_covers_text(self):
        text = "Alpha beta. " * 200 + "\n" + "X" * 2000
        chunks = R.chunk_source(text, 700)
        self.assertTrue(all(len(c) <= 700 for c in chunks))
        self.assertGreater(len(chunks), 3)

    def test_refuses_non_test_split(self):
        with self.assertRaises(SystemExit):
            R.main(["--split", "train", "--dry-run", "--fixture", str(FIXTURE)])

    @unittest.skipUnless(CKPT.exists(), "warm-start checkpoint not present")
    def test_dry_run_identical_inputs_and_budget_respected(self):
        with tempfile.TemporaryDirectory() as d:
            out = R.main(["--dry-run", "--fixture", str(FIXTURE), "--split", "any", "--allow-non-test",
                          "--n", "6", "--budgets", "1,3", "--linucb", f"LinUCB={CKPT}",
                          "--random-seeds", "0,1", "--out-dir", d, "--tag", "t"])
            res = out["results"]
            n_resp = out["manifest"]["protocol"]["n_responses"]
            self.assertEqual({r["policy"] for r in res},
                             {"LinUCB", "StaticCTSRanked", "StaticFIFO", "DeterministicCTS", "Random", "VerifyAll"})
            for r in res:
                self.assertLessEqual(r["recorded_cost"]["model_calls"], r["budget_model_calls"] * n_resp)
                self.assertEqual(r["n_responses"], n_resp)
            # tighter budget verifies no more claims than a looser one, per policy
            for name in {r["policy"] for r in res}:
                lo = [r for r in res if r["policy"] == name and r["budget_model_calls"] == 1 and r["seed"] == 0]
                hi = [r for r in res if r["policy"] == name and r["budget_model_calls"] == 3 and r["seed"] == 0]
                if lo and hi:
                    self.assertLessEqual(lo[0]["n_claims_verified"], hi[0]["n_claims_verified"])
            # guard against silently scoring nothing (e.g. gold-label case mismatch)
            for r in res:
                self.assertGreater(r["claim_broad"]["tp"] + r["claim_broad"]["fp"] + r["claim_broad"]["fn"]
                                   + r["claim_broad"]["tn"], 0, r["policy"])
            self.assertTrue(out["manifest"]["dry_run"])
            self.assertFalse(out["manifest"]["components"]["assessor"]["real"])
            self.assertTrue(list(Path(d).glob("*.md")))


class _OracleAssessor:
    """Deterministic verifier that knows the gold label (test-only): flags exactly the gold-hallucinated claims."""
    assessor_id = "oracle"

    def __init__(self, gold_by_text):
        self.gold_by_text = gold_by_text

    def assess(self, claim, passages):
        from src.evidence.models import AssessmentLabel, EvidenceAssessment, VerificationCost
        bad = self.gold_by_text.get(claim.normalized_text) == "HALLUCINATED"
        return EvidenceAssessment(
            claim_id=claim.claim_id, evidence_ids=[p.evidence_id for p in passages], supporting_spans={},
            label=AssessmentLabel.CONTRADICTED if bad else AssessmentLabel.SUPPORTED, confidence=0.9,
            rationale="oracle", cost=VerificationCost(model_calls=1, latency_seconds=0.0),
        )


class TestGoldLabelMappingAndScoring(unittest.TestCase):
    def _prepared(self):
        from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter
        ad = RAGTruthDatasetAdapter()
        ad.load_from_fixture(FIXTURE)
        return R.prepare(ad, ad.responses)[0]

    def test_gold_labels_are_canonical_uppercase_and_include_positives(self):
        gold = {g for p in self._prepared() for g in p["gold"].values()}
        self.assertTrue(gold <= {"HALLUCINATED", "UNFLAGGED", "UNCERTAIN", "IMPLICIT_TRUE_ABSENT", "MIXED"}, gold)
        self.assertIn("HALLUCINATED", gold)  # fixture must exercise the positive class
        self.assertIn("UNFLAGGED", gold)

    def test_oracle_verifier_scores_perfectly_through_real_scoring_path(self):
        prepared = self._prepared()
        by_text = {}
        for p in prepared:
            for c in p["claims"]:
                by_text.setdefault(c.normalized_text, set()).add(p["gold"][c.claim_id])
        self.assertTrue(all(len(v) == 1 for v in by_text.values()), "fixture has ambiguous duplicate claim texts")
        oracle = _OracleAssessor({t: next(iter(v)) for t, v in by_text.items()})
        cache = R.LLMCache(None)
        res, rows = R.run_setting("VerifyAll", 50, 0, prepared, {}, R.CachedAssessor(oracle, cache),
                                  R.CachedRepairer(R._CostedMockRepairer(), cache), cache)
        n_pos = sum(1 for r in rows if r["gold"] == "HALLUCINATED")
        self.assertGreater(n_pos, 0)
        for mode in ("strict", "broad"):
            c = res[f"claim_{mode}"]
            self.assertEqual(c["tp"], n_pos)
            self.assertEqual((c["fp"], c["fn"]), (0, 0))
            self.assertEqual((c["precision"], c["recall"], c["f1"]), (1.0, 1.0, 1.0))

    def test_blind_verifier_scores_zero_not_none(self):
        prepared = self._prepared()
        cache = R.LLMCache(None)
        oracle = _OracleAssessor({})  # never flags
        res, rows = R.run_setting("StaticFIFO", 50, 0, prepared, {}, R.CachedAssessor(oracle, cache),
                                  R.CachedRepairer(R._CostedMockRepairer(), cache), cache)
        c = res["claim_broad"]
        self.assertEqual(c["tp"], 0)
        self.assertGreater(c["fn"], 0)
        self.assertEqual(c["recall"], 0.0)

    def test_unverified_hallucinated_claims_are_false_negatives(self):
        prepared = self._prepared()
        cache = R.LLMCache(None)
        res, rows = R.run_setting("VerifyAll", 0, 0, prepared, {}, R.CachedAssessor(_OracleAssessor({}), cache),
                                  R.CachedRepairer(R._CostedMockRepairer(), cache), cache)
        self.assertEqual(res["n_claims_verified"], 0)
        n_pos = sum(1 for r in rows if r["gold"] == "HALLUCINATED")
        self.assertEqual(res["claim_broad"]["fn"], n_pos)



class TestOracleSelectionControl(unittest.TestCase):
    """--verifier oracle: gold-label control that isolates policy SELECTION from verifier error."""

    def _run(self, budgets="1,2,50", extra=()):
        with tempfile.TemporaryDirectory() as d:
            return R.main(["--dry-run", "--verifier", "oracle", "--fixture", str(FIXTURE), "--split", "any", "--allow-non-test",
                           "--all-responses", "--budgets", budgets, "--random-seeds", "0,1", "--n-boot", "50",
                           "--policies", "StaticCTSRanked,StaticFIFO,DeterministicCTS,Random,VerifyAll",
                           "--out-dir", d, "--tag", "o", *extra])

    def test_oracle_control_properties(self):
        out = self._run()
        man = out["manifest"]
        self.assertEqual(man["protocol"]["verifier"], "oracle")
        self.assertFalse(man["protocol"]["repair_enabled"])
        self.assertFalse(man["components"]["assessor"]["real"])
        self.assertIn("not a verifier", man["components"]["assessor"]["note"])
        n_resp = man["protocol"]["n_responses"]
        for r in out["results"]:
            self.assertNotIn("REPAIR", r["action_counts"])                                   # repair disabled
            self.assertLessEqual(r["recorded_cost"]["model_calls"], r["budget_model_calls"] * n_resp)  # budget respected
            for mode in ("strict", "broad"):
                self.assertEqual(r[f"claim_{mode}"]["fp"], 0)                                 # oracle never false-flags
            # detection by an oracle == selection: every verified positive is flagged
            self.assertEqual(r["claim_broad"]["tp"], r["selection"]["verified_positives"])
            self.assertEqual(r["claim_broad"]["recall"], r["selection"]["selection_recall"])

    def test_unconstrained_budget_selects_everything_and_scores_perfectly(self):
        out = self._run(budgets="50")
        verify_all = [r for r in out["results"] if r["policy"] == "VerifyAll"][0]
        self.assertEqual(verify_all["selection"]["selection_recall"], 1.0)
        self.assertEqual(verify_all["claim_broad"]["f1"], 1.0)
        self.assertEqual(verify_all["selection"]["coverage"], 1.0)

    def test_tighter_budget_never_selects_more(self):
        out = self._run(budgets="1,3")
        for name in {r["policy"] for r in out["results"]}:
            lo = [r for r in out["results"] if r["policy"] == name and r["budget_model_calls"] == 1 and r["seed"] == 0][0]
            hi = [r for r in out["results"] if r["policy"] == name and r["budget_model_calls"] == 3 and r["seed"] == 0][0]
            self.assertLessEqual(lo["n_claims_verified"], hi["n_claims_verified"])

    def test_all_responses_uses_the_whole_split_and_default_mode_is_unchanged(self):
        out = self._run(budgets="2")
        self.assertEqual(out["manifest"]["protocol"]["sampling"], "all responses of the split")
        from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter
        ad = RAGTruthDatasetAdapter()
        ad.load_from_fixture(FIXTURE)
        expected = len(R.prepare(ad, [r for r in ad.responses if r.source_id in ad.sources])[0])  # responses with >= 1 claim
        self.assertGreater(expected, 0)
        self.assertEqual(out["manifest"]["protocol"]["n_responses"], expected)
        with tempfile.TemporaryDirectory() as d:  # default llm path (mock components via --dry-run) still reports verifier=llm
            o2 = R.main(["--dry-run", "--fixture", str(FIXTURE), "--split", "any", "--allow-non-test", "--n", "3", "--budgets", "2",
                         "--linucb", f"LinUCB={CKPT}", "--random-seeds", "0", "--out-dir", d, "--tag", "d"]) if CKPT.exists() else None
            if o2:
                self.assertEqual(o2["manifest"]["protocol"]["verifier"], "llm")
                self.assertTrue(o2["manifest"]["protocol"]["repair_enabled"])
                self.assertIn("selection", o2["results"][0])  # new selection block present in every setting


if __name__ == "__main__":
    unittest.main()
