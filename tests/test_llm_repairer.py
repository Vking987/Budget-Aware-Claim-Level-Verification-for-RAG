"""Unit tests for LLMRepairer (src/controller/interfaces.py).

Tests three execution paths:
1. Successful LLM response with valid JSON.
2. Empty/unparseable JSON -> deterministic fallback revision.
3. Server exception -> MockRepairer fallback with error-tagged reason.

No real server or Qdrant required.
"""

from __future__ import annotations

import json
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.claims.models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus
from src.controller.interfaces import LLMRepairer
from src.evidence.models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost


def _make_claim(text: str = "The capital of France is Paris.") -> ClaimRecord:
    return ClaimRecord(
        claim_id="c_test",
        source_sentence_id="s_0",
        original_text=text,
        normalized_text=text,
        claim_type=ClaimType.FACTUAL_RELATION,
        decomposition_status=DecompositionStatus.ATOMIC,
        state=ClaimState.SELECTED,
    )


def _make_assessment(label: AssessmentLabel = AssessmentLabel.CONTRADICTED) -> EvidenceAssessment:
    return EvidenceAssessment(
        claim_id="c_test",
        evidence_ids=["e1"],
        supporting_spans={},
        label=label,
        confidence=0.9,
        rationale="Evidence says Berlin is the capital of Germany.",
        cost=VerificationCost(model_calls=1),
    )


def _make_evidence() -> List[EvidencePassage]:
    return [
        EvidencePassage(
            evidence_id="e1",
            doc_title="World Capitals",
            source_text="The capital of Germany is Berlin. France capital is Paris.",
            retrieval_score=0.92,
        )
    ]


def _make_mock_client(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    mock_resp.usage.prompt_tokens = 50
    mock_resp.usage.completion_tokens = 30
    mock_resp.usage.total_tokens = 80
    client = MagicMock()
    client.chat.completions.create.return_value = mock_resp
    return client


class TestLLMRepairerSuccess:
    def test_valid_json_produces_repair_result(self):
        payload = json.dumps({
            "revised_claim": "The capital of Germany is Berlin.",
            "repair_reason": "Corrected country based on evidence.",
            "confidence": 0.88,
        })
        client = _make_mock_client(payload)
        repairer = LLMRepairer(client=client, model_name="test_model")
        result, cost = repairer.repair(_make_claim(), _make_assessment(), _make_evidence())
        assert result.claim_id == "c_test"
        assert "Berlin" in result.proposed_revision
        assert "Corrected" in result.repair_reason
        assert abs(result.confidence - 0.88) < 0.01
        assert cost.model_calls == 1

    def test_evidence_ids_propagated(self):
        payload = json.dumps({"revised_claim": "X.", "repair_reason": "Y.", "confidence": 0.5})
        client = _make_mock_client(payload)
        repairer = LLMRepairer(client=client, model_name="test_model")
        result, _ = repairer.repair(_make_claim(), _make_assessment(), _make_evidence())
        assert "e1" in result.supporting_evidence_ids


class TestLLMRepairerFallback:
    def test_empty_revised_claim_produces_empty_revision(self):
        payload = json.dumps({"revised_claim": "", "repair_reason": "Could not repair.", "confidence": 0.5})
        client = _make_mock_client(payload)
        repairer = LLMRepairer(client=client, model_name="test_model")
        result, cost = repairer.repair(_make_claim(), _make_assessment(), _make_evidence())
        # Does not fabricate a repaired claim
        assert result.proposed_revision == ""
        assert "LLM repair generation failed" in result.repair_reason
        assert result.confidence == 0.0
        assert cost.model_calls == 1

    def test_garbage_json_does_not_fabricate_claim(self):
        client = _make_mock_client("This is not JSON at all.")
        repairer = LLMRepairer(client=client, model_name="test_model")
        result, _ = repairer.repair(_make_claim(), _make_assessment(), _make_evidence())
        assert result.claim_id == "c_test"
        assert result.proposed_revision == ""
        assert "Parse failure" in result.repair_reason
        assert result.confidence == 0.0


class TestLLMRepairerServerError:
    def test_server_exception_raises_runtime_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = ConnectionError("llama.cpp server not responding")
        repairer = LLMRepairer(client=client, model_name="test_model")
        with pytest.raises(RuntimeError, match="LLMRepairer call failed.*ConnectionError"):
            repairer.repair(_make_claim(), _make_assessment(), _make_evidence())

    def test_client_init_failure_raises_runtime_error(self):
        repairer = LLMRepairer(base_url="http://127.0.0.1:9999/v1")
        with patch("openai.OpenAI", side_effect=ConnectionError("Cannot connect")):
            with pytest.raises(RuntimeError, match="LLMRepairer failed to connect"):
                repairer.repair(_make_claim(), _make_assessment(), _make_evidence())
