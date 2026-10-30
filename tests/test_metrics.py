"""Unit tests for evaluation metrics module with hand-checked reference values using standard unittest."""

import unittest
from src.evaluation.metrics import (
    normalize_answer,
    compute_exact_match,
    compute_token_f1,
    hit_at_k,
    recall_at_k,
    both_at_k,
    compute_retrieval_metrics,
    compute_generation_metrics,
)


class TestMetrics(unittest.TestCase):

    # ==============================================================================
    # Text Normalization & Generation Tests
    # ==============================================================================

    def test_normalization_removes_punctuation_articles_and_lowercases(self):
        self.assertEqual(normalize_answer("The Radio City!"), "radio city")
        self.assertEqual(normalize_answer("a quick, brown fox."), "quick brown fox")
        self.assertEqual(normalize_answer("An Apple"), "apple")
        self.assertEqual(normalize_answer("  extra   spaces  "), "extra spaces")
        self.assertEqual(normalize_answer(None), "")

    def test_exact_match_hand_checked_cases(self):
        self.assertEqual(compute_exact_match("Radio City", "radio city"), 1.0)
        self.assertEqual(compute_exact_match("The 1984 Olympic Games.", "1984 Olympic Games"), 1.0)
        self.assertEqual(compute_exact_match("Paris, France", "Paris France"), 1.0)
        self.assertEqual(compute_exact_match("Abraham Thomas", "Thomas Abraham"), 0.0)  # Word order matters for EM
        self.assertEqual(compute_exact_match("insufficient information", "yes"), 0.0)
        self.assertEqual(compute_exact_match("", ""), 1.0)
        self.assertEqual(compute_exact_match(None, None), 1.0)
        self.assertEqual(compute_exact_match("hello", None), 0.0)

    def test_token_f1_hand_checked_cases(self):
        # Identical after normalization -> F1 = 1.0
        self.assertEqual(compute_token_f1("Radio City", "radio city"), 1.0)

        # Permuted words -> F1 = 1.0 (Token F1 is bag-of-words)
        self.assertEqual(compute_token_f1("Abraham Thomas", "Thomas, Abraham"), 1.0)

        # Partial overlap:
        # Pred: "George Washington University" (3 tokens)
        # Gold: "George Washington" (2 tokens)
        # Common: 2 tokens. Precision = 2/3, Recall = 2/2 = 1.0. F1 = 2 * (2/3) * 1 / (2/3 + 1) = 4/3 / 5/3 = 4/5 = 0.8
        self.assertAlmostEqual(compute_token_f1("George Washington University", "George Washington"), 0.8, places=4)

        # Zero overlap -> F1 = 0.0
        self.assertEqual(compute_token_f1("New York", "Los Angeles"), 0.0)

        # Empty inputs
        self.assertEqual(compute_token_f1("", ""), 1.0)
        self.assertEqual(compute_token_f1("something", ""), 0.0)
        self.assertEqual(compute_token_f1("", "something"), 0.0)

    # ==============================================================================
    # Document Retrieval Metric Tests
    # ==============================================================================

    def test_retrieval_metrics_distinction_hit_recall_both(self):
        gold = {"Doc A", "Doc B"}
        retrieved = ["Doc C", "Doc A", "Doc D", "Doc B", "Doc E"]

        # At K=1 (retrieved: ["Doc C"]):
        # Neither gold document found
        self.assertEqual(hit_at_k(retrieved, gold, 1), 0.0)
        self.assertEqual(recall_at_k(retrieved, gold, 1), 0.0)
        self.assertEqual(both_at_k(retrieved, gold, 1), 0.0)

        # At K=2 (retrieved: ["Doc C", "Doc A"]):
        # Hit: Doc A is found -> 1.0
        # Recall: 1 out of 2 found -> 0.5
        # Both: Doc B is missing -> 0.0
        self.assertEqual(hit_at_k(retrieved, gold, 2), 1.0)
        self.assertEqual(recall_at_k(retrieved, gold, 2), 0.5)
        self.assertEqual(both_at_k(retrieved, gold, 2), 0.0)

        # At K=3 (retrieved: ["Doc C", "Doc A", "Doc D"]):
        # Still only Doc A found
        self.assertEqual(hit_at_k(retrieved, gold, 3), 1.0)
        self.assertEqual(recall_at_k(retrieved, gold, 3), 0.5)
        self.assertEqual(both_at_k(retrieved, gold, 3), 0.0)

        # At K=4 (retrieved: ["Doc C", "Doc A", "Doc D", "Doc B"]):
        # Both Doc A and Doc B found
        self.assertEqual(hit_at_k(retrieved, gold, 4), 1.0)
        self.assertEqual(recall_at_k(retrieved, gold, 4), 1.0)
        self.assertEqual(both_at_k(retrieved, gold, 4), 1.0)

    def test_retrieval_edge_cases(self):
        # Empty gold titles
        self.assertEqual(hit_at_k(["Doc A"], set(), 1), 0.0)
        self.assertEqual(recall_at_k(["Doc A"], set(), 1), 0.0)
        self.assertEqual(both_at_k(["Doc A"], set(), 1), 0.0)

        # Empty retrieved titles
        self.assertEqual(hit_at_k([], {"Doc A"}, 5), 0.0)
        self.assertEqual(recall_at_k([], {"Doc A"}, 5), 0.0)
        self.assertEqual(both_at_k([], {"Doc A"}, 5), 0.0)

        # Cutoff K <= 0
        self.assertEqual(hit_at_k(["Doc A"], {"Doc A"}, 0), 0.0)
        self.assertEqual(recall_at_k(["Doc A"], {"Doc A"}, 0), 0.0)
        self.assertEqual(both_at_k(["Doc A"], {"Doc A"}, 0), 0.0)

    def test_compute_retrieval_metrics_bundle(self):
        gold = {"Doc 1", "Doc 2"}
        retrieved = ["Doc 1", "Doc 3", "Doc 2", "Doc 4"]
        res = compute_retrieval_metrics(retrieved, gold, cutoffs=(1, 3, 5))

        self.assertEqual(res["hit_1"], 1.0)
        self.assertEqual(res["recall_1"], 0.5)
        self.assertEqual(res["both_1"], 0.0)

        self.assertEqual(res["hit_3"], 1.0)
        self.assertEqual(res["recall_3"], 1.0)
        self.assertEqual(res["both_3"], 1.0)


if __name__ == "__main__":
    unittest.main()
