"""Live relevance check: retrieval -> relevance gate -> grounded draft -> decomposition -> on-topic filter, for several questions (no verification).

    python -m webapp.check_questions [--source wikipedia] [--out experiments/webapp_runs/relevance_check.json] "question 1" "question 2" ...

Needs internet (Wikipedia) and the llama-server. Prints, per question, the search queries, kept and dropped pages, retrieved passages, the draft,
and which claims stayed or were set aside as off-topic. Nothing is hard-coded per question.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.claims.models import ClaimRecord, ClaimType, DecompositionStatus
from webapp.generation import generate_draft, is_insufficient
from webapp.relevance import analyse_question, claim_is_on_topic, passages_support_question
from webapp.runner import Backend, extract_claims, passage_url

DEFAULT_QUESTIONS = ["Who founded VIT and when was it established?", "Who wrote Frankenstein and when was it first published?",
                     "What caused the fall of the Western Roman Empire?", "Who created VIT University and when?"]


def check(backend: Backend, question: str, source: str = "wikipedia", top_k: int = 5) -> dict:
    t0 = time.monotonic()
    out = {"question": question, "source": source}
    analysis = analyse_question(question)
    out["analysis"] = analysis.to_dict()
    retr = backend.retriever(source)
    dummy = ClaimRecord(claim_id="q", source_sentence_id="s", original_text=question, normalized_text=question, claim_type=ClaimType.FACTUAL_RELATION,
                        decomposition_status=DecompositionStatus.ATOMIC)
    passages, _ = retr.retrieve(dummy, top_k=top_k)
    gate = passages_support_question(analysis, passages)
    passages = gate["kept"]
    out["trace"] = getattr(retr, "last_trace", None)
    out["passages"] = [{"title": p.doc_title, "url": passage_url(p)[0], "score": p.retrieval_score, "section": (p.metadata or {}).get("section"), "text": p.source_text[:240]} for p in passages]
    if not passages:
        out.update(outcome="insufficient_evidence", reason=gate["reason"] or "no passages retrieved")
        return out
    draft, _ = generate_draft(question, passages)
    out["draft"] = draft
    if is_insufficient(draft):
        out.update(outcome="insufficient_evidence", reason="generator reported the passages do not contain the answer")
        return out
    claims = extract_claims(backend.pipeline(), draft)
    kept, off = [], []
    for i, c in enumerate(claims):
        (kept if claim_is_on_topic(analysis, c.normalized_text, is_first=(i == 0)) else off).append(c.normalized_text)
    out.update(outcome="answered", claims_kept=kept, claims_off_topic=off, seconds=round(time.monotonic() - t0, 1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="*")
    ap.add_argument("--source", default="wikipedia")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    b = Backend()
    results = []
    for q in (a.questions or DEFAULT_QUESTIONS):
        try:
            r = check(b, q, a.source)
        except Exception as e:  # keep going: one failing question must not hide the rest
            r = {"question": q, "outcome": "error", "reason": f"{type(e).__name__}: {e}"}
        results.append(r)
        print(f"\n=== {q}\n  outcome: {r['outcome']}  {r.get('reason', '')}")
        if r.get("trace"):
            print(f"  queries: {r['trace']['queries']}\n  kept pages: {r['trace']['kept_pages'][:6]}\n  dropped: {[(d['title'], d['reason'][:28]) for d in r['trace']['dropped_pages']][:6]}")
        for p in r.get("passages", []):
            print(f"  passage [{p['title']} / {p['section']}] {p['text'][:100]!r}")
        if r.get("draft"):
            print(f"  draft: {r['draft']}")
            print(f"  claims kept: {r['claims_kept']}\n  off-topic:   {r['claims_off_topic']}")
        time.sleep(2.0)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        print("\nwrote", a.out)
    return results


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
