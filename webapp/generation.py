"""Grounded draft generation for the web demo.

The repository's pipeline prompt asks for "4 to 6 sentences", which makes a 1B model pad the answer with whatever else is in the context
(that is how unrelated campus and music claims entered an answer about VIT). This prompt asks for a direct answer first, a few supporting
sentences that bear on the question, and an explicit way out (INSUFFICIENT_CONTEXT) when the passages do not contain the answer.
The pipeline's own `_generate_draft` is unchanged and still used by the CLI.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, List, Optional, Tuple

from src.evidence.models import EvidencePassage, VerificationCost

INSUFFICIENT = "INSUFFICIENT_CONTEXT"
MAX_PASSAGE_CHARS = 650


def build_prompt(question: str, passages: List[EvidencePassage]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {p.doc_title}: {p.source_text[:MAX_PASSAGE_CHARS]}" for i, p in enumerate(passages))
    return (
        "You answer a question using ONLY the numbered context passages.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        "Rules:\n"
        "1. Write 2 to 4 complete sentences of plain prose. No numbering, no bullets, no fragments; each sentence names its subject.\n"
        "2. The first sentence directly answers the question (name the person, place or date it asks for).\n"
        "3. Any further sentence adds a fact from the context that directly supports or completes that answer.\n"
        "4. Every sentence must help answer the question. Do not mention other topics, people, places or institutions from the context.\n"
        "5. Use only facts stated in the context. Do not guess or add outside knowledge.\n"
        f"6. If the context does not contain the answer, reply with exactly: {INSUFFICIENT}\n\n"
        "Example of the style (a different question): \"Who discovered radium and when?\" -> "
        "\"Marie and Pierre Curie discovered radium in 1898. They announced the discovery in December of that year.\"\n\n"
        "Answer:"
    )


_LIST_MARK = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•])\s+", re.M)


def clean_draft(text: str) -> str:
    """Remove list numbering / bullets the model sometimes adds despite the prompt, and join lines: '1.' must never become a claim."""
    text = _LIST_MARK.sub("", text)
    return " ".join(text.split())


def is_insufficient(draft: str) -> bool:
    return draft.strip().upper().startswith(INSUFFICIENT)


SENT_SPLIT = re.compile(r"(?<![A-Z]\.)(?<=[.!?])\s+")  # a single capital + period is an initial ('G. Viswanathan'), not a sentence end


def is_well_formed(text: str) -> bool:
    """A usable draft has at least one real sentence (6+ words) and no sentence of one or two words ('Mary Shelley.', '1818.')."""
    sents = [s for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    lens = [len(s.split()) for s in sents]
    return bool(lens) and max(lens) >= 6 and min(lens) >= 3


def extractive_draft(analysis, passages: List[EvidencePassage], n: int = 2) -> str:
    """Fallback when the model cannot write a usable draft: the n passage sentences that best answer the question, verbatim and in source order.
    Nothing is generated, so nothing can be invented; the UI labels this draft as extractive."""
    from webapp.relevance import asks_when, has_year
    when_q = asks_when(analysis.question)
    scored = []
    for pi, p in enumerate(passages):  # passages already passed the entity gate; a sentence need not repeat the entity's name
        for si, s in enumerate(x.strip() for x in SENT_SPLIT.split(p.source_text) if x.strip()):
            if len(s.split()) < 6:
                continue
            scored.append((analysis.relevance(p.doc_title, s) + (0.15 if when_q and has_year(s) else 0.0), -pi, -si, pi, si, s))
    scored.sort(reverse=True)
    top = sorted(scored[:n], key=lambda t: (t[3], t[4]))
    return " ".join(t[5] for t in top)


RETRY_NOTE = ("\n\nIMPORTANT: your previous reply was not written as complete sentences ({prev!r}). Reply with 2 to 4 complete sentences, each with a subject and a verb, "
              "for example: \"X founded Y in 1984. Y was established as a college.\"\n\nAnswer:")


def generate_draft(question: str, passages: List[EvidencePassage], client: Optional[Any] = None) -> Tuple[str, VerificationCost]:
    """One generation, plus ONE retry with a stricter instruction when the reply is a fragment. The returned cost covers every call made."""
    text, cost = _generate_once(question, passages, client, None)
    if is_insufficient(text) or is_well_formed(text):
        return text, cost
    text2, cost2 = _generate_once(question, passages, client, text)
    merged = VerificationCost(model_calls=(cost.model_calls or 0) + (cost2.model_calls or 0), prompt_tokens=(cost.prompt_tokens or 0) + (cost2.prompt_tokens or 0),
                              completion_tokens=(cost.completion_tokens or 0) + (cost2.completion_tokens or 0), total_tokens=(cost.total_tokens or 0) + (cost2.total_tokens or 0),
                              latency_seconds=(cost.latency_seconds or 0) + (cost2.latency_seconds or 0), model_id=cost.model_id)
    return (text2 if is_well_formed(text2) or is_insufficient(text2) else text), merged


def _generate_once(question: str, passages: List[EvidencePassage], client: Optional[Any], previous: Optional[str]) -> Tuple[str, VerificationCost]:
    from openai import OpenAI
    start = time.monotonic()
    client = client or OpenAI(base_url=os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1"), api_key="local", timeout=60.0)
    try:
        models = client.models.list()
        model_name = models.data[0].id if models.data else "local_model"
    except Exception:
        model_name = "local_model"  # llama-server's /v1/models shape differs from OpenAI's; it ignores the name
    prompt = build_prompt(question, passages)
    if previous is not None:
        prompt = prompt[: -len("\n\nAnswer:")] + RETRY_NOTE.format(prev=previous[:120])
    try:
        resp = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=300)
    except Exception as exc:
        raise RuntimeError(f"Draft generation failed against the LLM server: {exc}") from exc
    usage = getattr(resp, "usage", None)
    pt, ct = (getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0) if usage else (0, 0)
    text = clean_draft((resp.choices[0].message.content or "").strip())
    return text, VerificationCost(model_calls=1, prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct, latency_seconds=time.monotonic() - start, model_id=model_name)
