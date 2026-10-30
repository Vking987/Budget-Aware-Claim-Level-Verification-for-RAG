"""Tests for the ALCE additions: ingestion helpers/guard, long-answer draft mode, collection passthrough."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ingest_alce as I  # noqa: E402
from src.controller.interfaces import MockRepairer, MockRetriever  # noqa: E402
from src.controller.retriever import QdrantRetriever  # noqa: E402
from src.evidence.assessor import MockDeterministicAssessor  # noqa: E402
from src.pipeline import RAGVerificationPipeline  # noqa: E402


def _q(sid, docs):
    return {"sample_id": sid, "question": f"q{sid}", "docs": docs}


class TestIngestHelpers(unittest.TestCase):
    def test_dedup_by_doc_id_and_sample_ids_tracked(self):
        qs = [
            _q("a", [{"id": "1", "title": "T1", "text": "text one"}, {"id": "2", "title": "T2", "text": "text two"}]),
            _q("b", [{"id": "2", "title": "T2", "text": "text two"}, {"id": "3", "title": "T3", "text": ""}]),
        ]
        docs, st = I.build_documents(qs, 100)
        self.assertEqual(st, {"questions": 2, "passages_supplied": 4, "unique_passages": 2})
        by_id = {d["source_doc_id"]: d for d in docs}
        self.assertEqual(by_id["2"]["sample_ids"], ["a", "b"])  # empty-text passage "3" dropped

    def test_top_docs_cap(self):
        qs = [_q("a", [{"id": str(i), "title": "T", "text": f"t{i}"} for i in range(10)])]
        docs, st = I.build_documents(qs, 3)
        self.assertEqual(len(docs), 3)

    def test_refuses_non_alce_collection_names(self):
        for bad in ("hotpotqa_bge-large-en-v1.5", "hotpot_wiki", "anything"):
            with patch.object(sys, "argv", ["ingest_alce.py", "--file", "nonexistent.json", "--collection", bad, "--recreate"]):
                with self.assertRaises(SystemExit) as cm:
                    I.main()
                self.assertIn("alce_", str(cm.exception.code))


class TestLongAnswerDraft(unittest.TestCase):
    def _pipeline(self, long_answer):
        client = MagicMock()
        resp = MagicMock()
        resp.choices[0].message.content = "Draft."
        resp.usage.prompt_tokens, resp.usage.completion_tokens = 10, 5
        client.chat.completions.create.return_value = resp
        pl = RAGVerificationPipeline(retriever=MockRetriever(), assessor=MockDeterministicAssessor(), repairer=MockRepairer(),
                                     synthesizer=object(), llm_client=client, long_answer=long_answer)
        return pl, client

    def test_long_answer_prompt_and_token_budget(self):
        pl, client = self._pipeline(True)
        pl._generate_draft("Why is the sky blue?", [])
        kw = client.chat.completions.create.call_args.kwargs
        self.assertIn("4 to 6 sentences", kw["messages"][0]["content"])
        self.assertEqual(kw["max_tokens"], 360)

    def test_default_draft_is_unchanged_concise(self):
        pl, client = self._pipeline(False)
        pl._generate_draft("Why is the sky blue?", [])
        kw = client.chat.completions.create.call_args.kwargs
        self.assertIn("concisely", kw["messages"][0]["content"])
        self.assertEqual(kw["max_tokens"], 256)


class TestCollectionPassthrough(unittest.TestCase):
    def test_default_collection_is_hotpotqa_and_custom_is_respected(self):
        self.assertEqual(QdrantRetriever().collection_name, "hotpotqa_bge-large-en-v1.5")
        self.assertEqual(QdrantRetriever(collection_name="alce_asqa_bge-large-en-v1.5").collection_name,
                         "alce_asqa_bge-large-en-v1.5")

    def test_pipeline_builds_retriever_on_requested_collection(self):
        with patch("src.pipeline.LlamaAssessor"), patch("src.pipeline.LLMRepairer"), patch("src.pipeline.LLMAnswerSynthesizer"):
            pl = RAGVerificationPipeline(collection_name="alce_asqa_bge-large-en-v1.5")
            self.assertEqual(pl.retriever.collection_name, "alce_asqa_bge-large-en-v1.5")
            self.assertEqual(RAGVerificationPipeline().retriever.collection_name, "hotpotqa_bge-large-en-v1.5")

    def test_retriever_exposes_source_doc_id_from_payload(self):
        hit = MagicMock(id=7, score=0.9, payload={"title": "T", "text": "body text", "source_doc_id": "doc_42", "url": "http://x"})
        client = MagicMock()
        client.search.return_value = [hit]
        enc = MagicMock()
        enc.encode.return_value = [0.1, 0.2]
        r = QdrantRetriever(client=client, embedding_model=enc, collection_name="alce_asqa_bge-large-en-v1.5")
        from src.claims.models import ClaimRecord, ClaimType, DecompositionStatus
        claim = ClaimRecord(claim_id="c", source_sentence_id="s", original_text="q", normalized_text="q",
                            claim_type=ClaimType.FACTUAL_RELATION, decomposition_status=DecompositionStatus.ATOMIC)
        passages, _ = r.retrieve(claim)
        self.assertEqual(passages[0].metadata["source_doc_id"], "doc_42")
        self.assertEqual(passages[0].evidence_id, "qdrant_7")
        self.assertEqual(passages[0].metadata["collection"], "alce_asqa_bge-large-en-v1.5")


if __name__ == "__main__":
    unittest.main()
