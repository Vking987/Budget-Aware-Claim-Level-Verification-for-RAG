"""Tests for scripts/build_support_annotation.py and scripts/score_support_annotation.py (synthetic data; no model, no cache)."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_support_annotation as B  # noqa: E402
import score_support_annotation as S  # noqa: E402
from src.evidence.models import AssessmentLabel  # noqa: E402

VERDICT = {"a": AssessmentLabel.SUPPORTED, "b": AssessmentLabel.INSUFFICIENT, "c": AssessmentLabel.CONTRADICTED, "d": AssessmentLabel.CONFLICTED}


def _prep(n_q=6):
    prep = {}
    for q in range(n_q):
        prep[f"q{q}"] = {"question": f"Question {q}?", "passages": [{"evidence_id": "p1", "doc_title": "T", "source_text": "passage text", "retrieval_score": 0.5}],
                         "claims": [{"claim_id": f"c{k}", "normalized_text": f"claim {k} of {q}", "original_text": f"claim {k} of {q}"} for k in "abcd"]}
    return prep


def _lookup(c, ps):
    v = VERDICT[c["claim_id"][-1]]
    return SimpleNamespace(label=v, rationale="NLI max entailment=0.100, max contradiction=0.900 over 1 evidence window(s).")


class TestBuild(unittest.TestCase):
    def test_records_skip_conflicted_and_keep_three_strata(self):
        recs = B.build_records(_prep(), _lookup)
        self.assertEqual({r["verdict"] for r in recs}, {"SUPPORTED", "INSUFFICIENT", "CONTRADICTED"})
        self.assertEqual(len(recs), 18)
        self.assertEqual(recs[0]["nli_max_contradiction"], 0.9)

    def test_select_census_vs_stratified_and_reproducible(self):
        recs = B.build_records(_prep(), _lookup)
        self.assertEqual(len(B.select(recs, 150, 0)), 18)
        a, b = B.select(recs, 9, 3), B.select(recs, 9, 3)
        self.assertEqual([(r["sample_id"], r["claim_id"]) for r in a], [(r["sample_id"], r["claim_id"]) for r in b])
        self.assertEqual({v: sum(r["verdict"] == v for r in a) for v in ("SUPPORTED", "INSUFFICIENT", "CONTRADICTED")}, {"SUPPORTED": 3, "INSUFFICIENT": 3, "CONTRADICTED": 3})

    def test_sheet_is_blind_key_has_strategy_and_no_overwrite(self):
        recs = B.build_records(_prep(), _lookup)
        with tempfile.TemporaryDirectory() as td:
            sheet, key, counts = B.write_files(recs, Path(td), 0, {"census": True, "claims_with_cached_verdict": 18})
            text = sheet.read_text(encoding="utf-8-sig")
            for leak in ("SUPPORTED", "INSUFFICIENT", "CONTRADICTED", "S0", "S1", "entailment"):
                self.assertNotIn(leak, text)
            k = json.loads(key.read_text())
            self.assertEqual(len(k["items"]), 18)
            self.assertTrue(all(v["in_S1"] for v in k["items"].values() if v["nli_verdict"] == "INSUFFICIENT"))
            self.assertFalse(any(v["in_S0"] for v in k["items"].values() if v["nli_verdict"] != "SUPPORTED"))
            self.assertEqual(counts, {"SUPPORTED": 6, "INSUFFICIENT": 6, "CONTRADICTED": 6})
            with self.assertRaises(FileExistsError):
                B.write_files(recs, Path(td), 0, {})


class TestScore(unittest.TestCase):
    def _fill(self, td, labeller):
        recs = B.build_records(_prep(), _lookup)
        sheet, key, _ = B.write_files(recs, Path(td), 0, {"census": True, "claims_with_cached_verdict": 18})
        k = json.loads(key.read_text())["items"]
        rows = list(csv.DictReader(open(sheet, newline="", encoding="utf-8-sig")))
        with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                r["label"] = labeller(k[r["item_id"]]) or ""
                w.writerow(r)
        return sheet, key

    def test_no_labels_means_no_report(self):
        with tempfile.TemporaryDirectory() as td:
            sheet, key = self._fill(td, lambda v: None)
            self.assertIsNone(S.render(sheet, key, None, None))

    def test_rates_follow_the_human_labels_not_the_nli_verdicts(self):
        lab = lambda v: {"SUPPORTED": "supported", "INSUFFICIENT": "unsupported", "CONTRADICTED": "contradicted"}[v["nli_verdict"]]  # noqa: E731
        with tempfile.TemporaryDirectory() as td:
            sheet, key = self._fill(td, lab)
            text = S.render(sheet, key, None, None)
        self.assertIn("18 of 18 items labelled", text)
        self.assertIn("S1 conservative", text)
        self.assertIn("Evidence support is not real-world truth", text)
        # S1 = 6 supported + 6 unsupported claims -> 0.50 supported, S0 = 1.00 supported
        self.assertIn("| S0 current (SUPPORTED verdicts) | 6 | 1.00", text)
        self.assertIn("| S1 conservative (SUPPORTED + INSUFFICIENT) | 12 | 0.50", text)

    def test_invalid_label_rejected_and_partial_labels_warn(self):
        with tempfile.TemporaryDirectory() as td:
            sheet, key = self._fill(td, lambda v: "supported" if v["nli_verdict"] == "SUPPORTED" else None)
            self.assertIn("WARNING: 12 items unlabelled", S.render(sheet, key, None, None))
        with tempfile.TemporaryDirectory() as td:
            sheet, key = self._fill(td, lambda v: "maybe")
            with self.assertRaises(ValueError):
                S.read_labels(sheet)

    def test_wilson_and_kappa(self):
        lo, hi = S.wilson(5, 10)
        self.assertTrue(lo < 0.5 < hi)
        self.assertIsNone(S.wilson(0, 0))
        self.assertAlmostEqual(S.cohen_kappa(["supported", "unsupported"] * 5, ["supported", "unsupported"] * 5), 1.0)
        self.assertLess(S.cohen_kappa(["supported", "unsupported"] * 5, ["unsupported", "supported"] * 5), 0)


if __name__ == "__main__":
    unittest.main()
