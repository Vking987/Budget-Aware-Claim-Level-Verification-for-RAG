import unittest

from src.evaluation.ragtruth_metrics import (
    bootstrap_ci, claim_counts_by_response, pooled, prf, repair_summary, response_level,
)


def _row(rid, gold, label):
    return {"response_id": rid, "gold": gold, "label": label}


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.rows = [
            _row("r1", "HALLUCINATED", "CONTRADICTED"),   # strict TP, broad TP
            _row("r1", "HALLUCINATED", "INSUFFICIENT"),   # strict FN, broad TP
            _row("r1", "HALLUCINATED", None),             # unverified -> FN both
            _row("r2", "UNFLAGGED", "SUPPORTED"),         # TN
            _row("r2", "UNFLAGGED", "INSUFFICIENT"),      # strict TN, broad FP
            _row("r2", "UNFLAGGED", "CONFLICTED"),        # FP both
            _row("r2", "UNCERTAIN", "CONTRADICTED"),      # excluded
        ]
        self.counts = claim_counts_by_response(self.rows)

    def test_strict_and_broad_counts(self):
        s = pooled(self.counts.values(), "strict")
        b = pooled(self.counts.values(), "broad")
        self.assertEqual((s["tp"], s["fp"], s["fn"], s["tn"]), (1, 1, 2, 2))
        self.assertEqual((b["tp"], b["fp"], b["fn"], b["tn"]), (2, 2, 1, 1))
        self.assertAlmostEqual(s["precision"], 0.5)
        self.assertAlmostEqual(s["recall"], 1 / 3)

    def test_excluded_gold_not_scored(self):
        total = sum(sum(v[k] for k in v if k.startswith("strict_")) for v in self.counts.values())
        self.assertEqual(total, 6)

    def test_f1_zero_not_none_when_both_zero_numerators(self):
        self.assertEqual(prf(0, 3, 4)["f1"], 0.0)
        self.assertIsNone(prf(0, 0, 0)["f1"])

    def test_response_level(self):
        rows = [
            {"gold": True, "labels": ["SUPPORTED", "CONTRADICTED"]},
            {"gold": True, "labels": [None]},
            {"gold": False, "labels": ["SUPPORTED"]},
            {"gold": False, "labels": ["CONFLICTED"]},
        ]
        m = response_level(rows, "strict")
        self.assertEqual((m["tp"], m["fn"], m["tn"], m["fp"]), (1, 1, 1, 1))

    def test_bootstrap_is_seeded_and_brackets_point_estimate(self):
        ci1 = bootstrap_ci(self.counts, "broad", "recall", n_boot=200, seed=1)
        ci2 = bootstrap_ci(self.counts, "broad", "recall", n_boot=200, seed=1)
        self.assertEqual(ci1, ci2)

    def test_repair_summary(self):
        ev = [
            {"gold": "HALLUCINATED", "revision_nonempty": True, "reverified": True},
            {"gold": "HALLUCINATED", "revision_nonempty": True, "reverified": False},
            {"gold": "UNFLAGGED", "revision_nonempty": False, "reverified": None},
        ]
        s = repair_summary(ev)
        self.assertEqual(s["repairs_triggered"], 3)
        self.assertEqual(s["reverified_pass"], 1)
        self.assertEqual(s["reverification_attempted"], 2)
        self.assertEqual(s["reverified_pass_on_gold_hallucinated"], 1)



class TestCtsTriageF1(unittest.TestCase):
    """compute_claim_evaluation_metrics: F1 is 0.0 (not None) when precision and recall are defined but zero."""

    def _mapped(self):
        from pathlib import Path
        from src.evaluation.ragtruth_adapter import ClaimAnnotationLabel, RAGTruthDatasetAdapter
        ad = RAGTruthDatasetAdapter()
        ad.load_from_fixture(Path(__file__).resolve().parent / "fixtures" / "ragtruth_sample.json")
        return ad, ad.process_all(), ClaimAnnotationLabel

    def test_zero_precision_and_recall_gives_zero_f1_not_none(self):
        from src.evaluation.ragtruth_adapter import compute_claim_evaluation_metrics
        ad, mapped, L = self._mapped()
        for m in mapped:  # CTS ranks every positive low and every negative high
            m.triage_score = 0.1 if m.mapped_label == L.HALLUCINATED else 0.9
        res = compute_claim_evaluation_metrics(mapped, ad.responses, cts_threshold=0.5)
        self.assertEqual(res.cts_triage_precision, 0.0)
        self.assertEqual(res.cts_triage_recall, 0.0)
        self.assertEqual(res.cts_triage_f1, 0.0)

    def test_undefined_when_nothing_is_predicted_positive(self):
        from src.evaluation.ragtruth_adapter import compute_claim_evaluation_metrics
        ad, mapped, L = self._mapped()
        for m in mapped:
            m.triage_score = 0.0
        res = compute_claim_evaluation_metrics(mapped, ad.responses, cts_threshold=0.5)
        self.assertIsNone(res.cts_triage_precision)
        self.assertIsNone(res.cts_triage_f1)

    def test_perfect_ranking_gives_f1_one(self):
        from src.evaluation.ragtruth_adapter import compute_claim_evaluation_metrics
        ad, mapped, L = self._mapped()
        for m in mapped:
            m.triage_score = 0.9 if m.mapped_label == L.HALLUCINATED else 0.1
        res = compute_claim_evaluation_metrics(mapped, ad.responses, cts_threshold=0.5)
        self.assertEqual(res.cts_triage_f1, 1.0)



class TestSelectionMode(unittest.TestCase):
    """mode 'sel' separates SELECTION (was a claim verified at all) from verifier accuracy."""

    def test_selection_recall_precision_and_coverage(self):
        rows = [
            _row("r1", "HALLUCINATED", "SUPPORTED"),   # positive, verified (verifier wrong) -> selected
            _row("r1", "HALLUCINATED", None),          # positive never verified -> selection miss
            _row("r1", "UNFLAGGED", "SUPPORTED"),      # negative verified
            _row("r2", "UNFLAGGED", None),             # negative not verified
            _row("r2", "UNCERTAIN", "SUPPORTED"),      # excluded from scoring
        ]
        counts = claim_counts_by_response(rows)
        sel = pooled(counts.values(), "sel")
        self.assertEqual((sel["tp"], sel["fn"], sel["fp"], sel["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(sel["recall"], 0.5)          # selection recall
        self.assertAlmostEqual(sel["precision"], 0.5)       # verified claims that were gold positives
        self.assertAlmostEqual(sel["flagged_fraction"], 0.5)  # coverage of scored claims
        # the verifier metrics are independent of selection: the one verified positive was called SUPPORTED -> not detected
        self.assertEqual(pooled(counts.values(), "broad")["tp"], 0)

    def test_selection_ci_is_seeded(self):
        rows = [_row(f"r{i}", "HALLUCINATED", "SUPPORTED" if i % 2 else None) for i in range(40)]
        counts = claim_counts_by_response(rows)
        a = bootstrap_ci(counts, "sel", "recall", n_boot=200, seed=3)
        self.assertEqual(a, bootstrap_ci(counts, "sel", "recall", n_boot=200, seed=3))
        self.assertTrue(a[0] <= 0.5 <= a[1])


if __name__ == "__main__":
    unittest.main()
