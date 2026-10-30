"""Focused tests for scripts/run_ragtruth_verifier_diagnostic.py (no LLM)."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_ragtruth_comparison as R  # noqa: E402
import run_ragtruth_verifier_diagnostic as D  # noqa: E402
from src.evaluation import ragtruth_metrics as M  # noqa: E402
from src.evaluation.ragtruth_adapter import RAGTruthDatasetAdapter  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "ragtruth_sample.json"
PILOT = ROOT / "experiments" / "runs" / "ragtruth_comparison_20261028_162847_pilot50.json"

BIZ = {
    "name": "Apna Indian Kitchen", "address": "718 State St", "city": "Santa Barbara", "state": "CA",
    "categories": "Vegan, Indian", "hours": {"Tuesday": "16:30-21:0", "Sunday": "16:30-21:0"},
    "attributes": {"Music": None, "WiFi": "no", "OutdoorSeating": True, "Ambience": {"hipster": True, "romantic": None}},
    "business_stars": 4.5,
    "review_info": [{"review_stars": 5.0, "review_date": "2020-01-01", "review_text": "Great food."}],
}


def _src(task, info, sid="s1"):
    ns = SimpleNamespace(task_type=task, source_info=info, source=None, source_id=sid)
    ns.get_source_text = lambda: info if isinstance(info, str) else "\n".join(info.get("passages", []))
    return ns


class TestRendering(unittest.TestCase):
    def test_data2txt_is_readable_text_with_nulls_explicit(self):
        rec, rev = D.render_data2txt(BIZ)
        text = "\n".join(rec)
        self.assertIn("Music: not specified", text)
        self.assertIn("WiFi: no", text)
        self.assertIn("OutdoorSeating: yes", text)
        self.assertIn("Opening hours: Tuesday 16:30-21:0; Sunday 16:30-21:0", text)
        self.assertIn("Ambience: hipster = yes; romantic = not specified", text)
        self.assertIn("Overall rating: 4.5 stars", text)
        for ch in '{}"[]':
            self.assertNotIn(ch, text)
        self.assertEqual(rev, ["Review (5.0 stars): Great food."])

    def test_pack_lines_bounds_and_keeps_lines(self):
        lines = [f"Line {i}: " + "x" * 100 for i in range(40)] + ["Long. " * 400]
        chunks = D.pack_lines(lines)
        self.assertTrue(all(len(c) <= D.READABLE_CHUNK_CHARS for c in chunks))
        self.assertTrue(chunks[0].startswith("Line 0:"))

    def test_small_data2txt_is_fully_visible_record_first(self):
        ev, visible, total = D.readable_evidence(_src("Data2txt", BIZ), "The restaurant has no music.")
        self.assertTrue(visible)
        self.assertIn("Business name", ev[0].source_text)
        self.assertTrue(any("Great food" in p.source_text for p in ev))

    def test_large_data2txt_keeps_record_and_caps_passages(self):
        big = dict(BIZ)
        big["review_info"] = [{"review_stars": 3, "review_text": f"review {i} " + "word " * 80} for i in range(40)]
        ev, visible, total = D.readable_evidence(_src("Data2txt", big), "word review 7")
        self.assertFalse(visible)
        self.assertLessEqual(len(ev), D.MAX_PASSAGES)
        self.assertIn("Business name", ev[0].source_text)
        self.assertTrue(all(len(p.source_text) <= D.READABLE_CHUNK_CHARS for p in ev))
        self.assertGreater(total, sum(len(p.source_text) for p in ev))

    def test_text_sources_visibility(self):
        short, vis, _ = D.readable_evidence(_src("Summary", "A short article about cats. Cats sleep."), "Cats sleep")
        self.assertTrue(vis)
        long_text = "\n".join(f"Paragraph {i}. " + "filler " * 120 for i in range(12))
        ev, vis2, _ = D.readable_evidence(_src("QA", long_text), "Paragraph 5")
        self.assertFalse(vis2)
        self.assertLessEqual(len(ev), D.MAX_PASSAGES)
        self.assertTrue(any("Paragraph 5." in p.source_text for p in ev))  # lexical selection finds the relevant chunk


class _Oracle:
    assessor_id = "oracle"

    def __init__(self, positives):
        self.positives = positives

    def assess(self, claim, passages):
        from src.evidence.models import AssessmentLabel, EvidenceAssessment, VerificationCost
        bad = claim.normalized_text in self.positives
        return EvidenceAssessment(claim_id=claim.claim_id, evidence_ids=[p.evidence_id for p in passages], supporting_spans={},
                                  label=AssessmentLabel.CONTRADICTED if bad else AssessmentLabel.SUPPORTED, confidence=0.9,
                                  rationale="oracle", cost=VerificationCost(model_calls=1, latency_seconds=0.5))


class TestClaimSetAndScoring(unittest.TestCase):
    def setUp(self):
        self.ad = RAGTruthDatasetAdapter()
        self.ad.load_from_fixture(FIXTURE)
        self.prepared, self.mapped = R.prepare(self.ad, self.ad.responses)

    def test_label_mapping_validation_on_fixture(self):
        items = D.build_claim_set(self.prepared, self.mapped, 5, 0)
        v = D.validate_label_mapping(items)
        self.assertEqual(v["positives_without_overlapping_span"], [])
        self.assertEqual(v["negatives_with_overlapping_span"], [])
        self.assertGreater(sum(1 for i in items if i["gold"] == "HALLUCINATED"), 0)
        self.assertTrue(v["positive_label_type_counts"])

    def test_claim_set_is_deterministic_and_contains_all_positives(self):
        a = D.build_claim_set(self.prepared, self.mapped, 2, 7)
        b = D.build_claim_set(self.prepared, self.mapped, 2, 7)
        key = lambda xs: [(i["prep"]["resp"].response_id, i["claim"].claim_id) for i in xs]
        self.assertEqual(key(a), key(b))
        all_pos = sum(1 for p in self.prepared for g in p["gold"].values() if g == "HALLUCINATED")
        self.assertEqual(sum(1 for i in a if i["gold"] == "HALLUCINATED"), all_pos)

    def test_oracle_scores_perfectly_and_blind_scores_zero(self):
        items = D.build_claim_set(self.prepared, self.mapped, 50, 0)
        pos_texts = {i["claim"].normalized_text for i in items if i["gold"] == "HALLUCINATED"}
        neg_texts = {i["claim"].normalized_text for i in items if i["gold"] == "UNFLAGGED"}
        self.assertFalse(pos_texts & neg_texts, "fixture has ambiguous duplicate texts")
        for assessor, expect_tpr in ((_Oracle(pos_texts), 1.0), (_Oracle(set()), 0.0)):
            cache = R.LLMCache(None)
            wrapped = R.CachedAssessor(assessor, cache)
            res = {c: D.run_condition(items, c, wrapped, cache, self.ad.sources) for c in D.CONDITIONS}
            summ = D.summarize(items, res)
            for cond in D.CONDITIONS:
                for mode in ("strict", "broad"):
                    o = summ[cond]["overall"][mode]
                    self.assertEqual(o["recall"], expect_tpr)
                    self.assertEqual(o["fp"], 0)
                    self.assertEqual(o["fpr"], 0.0)
                self.assertAlmostEqual(summ[cond]["latency"]["mean_recorded_s"], 0.5)
            self.assertIn("paired_positives_broad", summ)

    def test_summary_splits_by_task_and_visibility(self):
        items = D.build_claim_set(self.prepared, self.mapped, 50, 0)
        cache = R.LLMCache(None)
        w = R.CachedAssessor(_Oracle(set()), cache)
        res = {c: D.run_condition(items, c, w, cache, self.ad.sources) for c in D.CONDITIONS}
        s = D.summarize(items, res)
        tasks = {i["task"] for i in items}
        self.assertEqual(set(s["A_original"]["by_task"]["broad"]), tasks)
        n = s["B_readable"]["full_source_visible_True"]["n_claims"] + s["B_readable"]["full_source_visible_False"]["n_claims"]
        self.assertEqual(n, len(items))


class TestWilson(unittest.TestCase):
    def test_known_values(self):
        lo, hi = M.wilson_ci(0, 17)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.1843, places=3)
        lo, hi = M.wilson_ci(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        self.assertIsNone(M.wilson_ci(0, 0))


@unittest.skipUnless(PILOT.exists() and (ROOT / "data/ragtruth/response.jsonl").exists(), "pilot run/data not present")
class TestEndToEndMock(unittest.TestCase):
    def test_dry_run_writes_separate_outputs_and_does_not_touch_benchmark_cache(self):
        cache_file = R.DEFAULT_CACHE
        before = cache_file.stat().st_mtime_ns if cache_file.exists() else None
        with tempfile.TemporaryDirectory() as d:
            out = D.main(["--dry-run", "--neg-per-task", "3", "--out-dir", d, "--tag", "t"])
            self.assertTrue(list(Path(d).glob("*.json")) and list(Path(d).glob("*.md")))
            self.assertTrue(out["meta"]["dry_run"])
            self.assertEqual(out["meta"]["n_pos"], 17)  # pilot sample has 17 gold-hallucinated claims
            self.assertEqual(out["meta"]["label_validation"]["positives_without_overlapping_span"], [])
        after = cache_file.stat().st_mtime_ns if cache_file.exists() else None
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
