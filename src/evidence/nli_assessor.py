"""NLI cross-encoder assessor: a drop-in alternative to LlamaAssessor.

Each (evidence window, claim) pair is scored by a sequence-pair NLI model (premise = evidence, hypothesis = claim). The claim label comes from
the strongest evidence window, with FIXED thresholds (not tuned on any benchmark):

    max P(entailment) >= entail_threshold                       -> SUPPORTED
    else max P(contradiction) >= contradict_threshold           -> CONTRADICTED
    else                                                         -> INSUFFICIENT

One claim assessment counts as one model call (windows are batched inside it); tokens and latency are recorded.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple

from src.claims.models import ClaimRecord

from .assessor import BaseAssessor
from .models import AssessmentLabel, EvidenceAssessment, EvidencePassage, VerificationCost

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def window_text(tokenizer: Any, text: str, max_tokens: int, stride: int) -> List[str]:
    """Split text into overlapping token windows of at most max_tokens (decoded back to strings). Short text is one window."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return [text]
    out, i = [], 0
    while i < len(ids):
        out.append(tokenizer.decode(ids[i:i + max_tokens]))
        if i + max_tokens >= len(ids):
            break
        i += stride
    return out


class NLIAssessor(BaseAssessor):
    def __init__(self, model_name: str = DEFAULT_MODEL, entail_threshold: float = 0.5, contradict_threshold: float = 0.5,
                 max_passages: int = 4, window_tokens: int = 300, stride_tokens: int = 200, batch_size: int = 8,
                 model: Optional[Any] = None, tokenizer: Optional[Any] = None, label_order: Optional[Tuple[str, ...]] = None) -> None:
        self.model_name = model_name
        self.entail_threshold, self.contradict_threshold = entail_threshold, contradict_threshold
        self.max_passages, self.window_tokens, self.stride_tokens, self.batch_size = max_passages, window_tokens, stride_tokens, batch_size
        self._model, self._tok = model, tokenizer
        self._order = label_order

    @property
    def assessor_id(self) -> str:
        return f"nli_assessor_{self.model_name.split('/')[-1]}"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
        id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
        self._order = tuple(id2label[i] for i in range(len(id2label)))
        if not {"entailment", "contradiction"} <= set(self._order):
            raise ValueError(f"model labels {self._order} lack entailment/contradiction")

    def _probs(self, premises: List[str], hypothesis: str):
        import torch
        out = []
        for i in range(0, len(premises), self.batch_size):
            batch = premises[i:i + self.batch_size]
            enc = self._tok(batch, [hypothesis] * len(batch), truncation="only_first", max_length=512, padding=True, return_tensors="pt")
            with torch.no_grad():
                out.append(torch.softmax(self._model(**enc).logits, dim=-1))
        return torch.cat(out), int(sum(1 for _ in premises))

    def assess(self, claim: ClaimRecord, evidence: List[EvidencePassage]) -> EvidenceAssessment:
        claim_text = claim.normalized_text or claim.original_text
        ids = [e.evidence_id for e in evidence]
        if not evidence:
            return EvidenceAssessment(claim_id=claim.claim_id, evidence_ids=[], supporting_spans={}, label=AssessmentLabel.INSUFFICIENT, confidence=0.95,
                                      rationale="No evidence passages provided for assessment.", limitations="Cannot assess without evidence.",
                                      cost=VerificationCost(model_calls=0), assessor_id=self.assessor_id)
        t0 = time.monotonic()
        try:
            self._load()
            windows: List[Tuple[str, str]] = []  # (evidence_id, text)
            for e in evidence[:self.max_passages]:
                for w in window_text(self._tok, e.source_text, self.window_tokens, self.stride_tokens):
                    windows.append((e.evidence_id, w))
            probs, _ = self._probs([w for _, w in windows], claim_text)
            ent_i, con_i = self._order.index("entailment"), self._order.index("contradiction")
            ent, con = probs[:, ent_i], probs[:, con_i]
            be, bc = int(ent.argmax()), int(con.argmax())
            ent_max, con_max = float(ent[be]), float(con[bc])
            if ent_max >= self.entail_threshold:
                label, conf, ev, txt = AssessmentLabel.SUPPORTED, ent_max, windows[be][0], windows[be][1]
            elif con_max >= self.contradict_threshold:
                label, conf, ev, txt = AssessmentLabel.CONTRADICTED, con_max, windows[bc][0], windows[bc][1]
            else:
                label, conf, ev, txt = AssessmentLabel.INSUFFICIENT, 1.0 - max(ent_max, con_max), None, None
            n_tokens = int(sum(len(self._tok(w, add_special_tokens=False)["input_ids"]) for _, w in windows))
            return EvidenceAssessment(
                claim_id=claim.claim_id, evidence_ids=ids, supporting_spans={ev: [txt[:300]]} if ev else {}, label=label, confidence=conf,
                rationale=f"NLI max entailment={ent_max:.3f}, max contradiction={con_max:.3f} over {len(windows)} evidence window(s).",
                cost=VerificationCost(model_calls=1, prompt_tokens=n_tokens, total_tokens=n_tokens, latency_seconds=time.monotonic() - t0, model_id=self.model_name),
                assessor_id=self.assessor_id)
        except Exception as e:  # mirror LlamaAssessor: a technical failure is an error assessment, never a silent verdict
            return EvidenceAssessment(claim_id=claim.claim_id, evidence_ids=ids, supporting_spans={}, label=AssessmentLabel.INSUFFICIENT, confidence=0.0,
                                      rationale=f"NLI call failed: {type(e).__name__}: {str(e)[:200]}", limitations="Assessment could not be completed.",
                                      cost=VerificationCost(model_calls=1, latency_seconds=time.monotonic() - t0, model_id=self.model_name),
                                      assessor_id=self.assessor_id, is_error=True, error_type=type(e).__name__)
