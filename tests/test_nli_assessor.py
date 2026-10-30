"""Tests for src/evidence/nli_assessor.py with a scripted NLI scorer (no model download)."""

import unittest

import torch

from src.claims.models import ClaimRecord, ClaimType, DecompositionStatus
from src.evidence.models import AssessmentLabel, EvidencePassage
from src.evidence.nli_assessor import NLIAssessor, window_text


class _Tok:
    """Whitespace 'tokenizer' with the two calls window_text/assess use."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


class _Scripted(NLIAssessor):
    """P(entail, neutral, contradict) from keywords in the window: 'yes' entails, 'no' contradicts."""

    def __init__(self, **kw):
        super().__init__(model=object(), tokenizer=_Tok(), label_order=("entailment", "neutral", "contradiction"), **kw)

    def _probs(self, premises, hypothesis):
        rows = [[0.9, 0.05, 0.05] if "yes" in p else [0.05, 0.05, 0.9] if "no" in p else [0.2, 0.6, 0.2] for p in premises]
        return torch.tensor(rows), len(premises)


def _claim():
    t = "The film was released in 1999."
    return ClaimRecord(claim_id="c", source_sentence_id="s", original_text=t, normalized_text=t, claim_type=ClaimType.FACTUAL_RELATION,
                       decomposition_status=DecompositionStatus.ATOMIC, triage_score=0.5)


def _ev(*texts):
    return [EvidencePassage(evidence_id=f"p{i}", doc_title="d", source_text=t, retrieval_score=0.5) for i, t in enumerate(texts)]


class TestNLIAssessor(unittest.TestCase):
    def test_labels_from_strongest_window(self):
        a = _Scripted()
        self.assertEqual(a.assess(_claim(), _ev("unrelated text", "yes it was")).label, AssessmentLabel.SUPPORTED)
        self.assertEqual(a.assess(_claim(), _ev("no it was not")).label, AssessmentLabel.CONTRADICTED)
        self.assertEqual(a.assess(_claim(), _ev("irrelevant text")).label, AssessmentLabel.INSUFFICIENT)

    def test_support_beats_contradiction_when_both_present(self):
        r = _Scripted().assess(_claim(), _ev("no it was not", "yes it was"))
        self.assertEqual(r.label, AssessmentLabel.SUPPORTED)
        self.assertEqual(list(r.supporting_spans), ["p1"])

    def test_cost_one_call_and_no_evidence_is_free_insufficient(self):
        a = _Scripted()
        r = a.assess(_claim(), _ev("yes", "no"))
        self.assertEqual(r.cost.model_calls, 1)
        self.assertGreater(r.cost.total_tokens, 0)
        e = a.assess(_claim(), [])
        self.assertEqual((e.label, e.cost.model_calls), (AssessmentLabel.INSUFFICIENT, 0))

    def test_failure_is_an_error_assessment_not_a_verdict(self):
        class Boom(_Scripted):
            def _probs(self, premises, hypothesis):
                raise RuntimeError("x")
        r = Boom().assess(_claim(), _ev("yes"))
        self.assertTrue(r.is_error)
        self.assertEqual(r.confidence, 0.0)

    def test_windowing_covers_long_text_with_overlap(self):
        tok = _Tok()
        text = " ".join(f"t{i}" for i in range(700))
        w = window_text(tok, text, 300, 200)
        self.assertEqual(len(w), 3)
        self.assertEqual(window_text(tok, "short text", 300, 200), ["short text"])

    def test_only_first_max_passages_are_scored(self):
        calls = []

        class Spy(_Scripted):
            def _probs(self, premises, hypothesis):
                calls.append(len(premises))
                return super()._probs(premises, hypothesis)
        Spy(max_passages=2).assess(_claim(), _ev("a", "b", "yes"))
        self.assertEqual(calls, [2])


if __name__ == "__main__":
    unittest.main()
