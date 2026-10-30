"""Post-hoc citation check: does every inline [Title] marker resolve to evidence the run actually retrieved?

The synthesizer writes inline citations as [Document Title]. Nothing upstream guarantees those titles are
real, so this module resolves each marker against the passages retrieved during the run (initial retrieval
plus any per-claim retrieval in the controller trace) and reports markers that resolve to nothing.
Title matching is case-insensitive, whitespace-normalised and HTML-entity-unescaped; a title shared by
several passages maps to all of them (ALCE/Wikipedia articles contribute multiple passages).
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, Iterable, List

_MARKER = re.compile(r"\[([^\[\]\n]{1,200})\]")


def _norm(s: str) -> str:
    return " ".join(html.unescape(s or "").lower().split())


def extract_markers(text: str) -> List[str]:
    """Distinct bracketed markers in first-seen order. Purely numeric markers like [1] are kept."""
    seen: List[str] = []
    for m in _MARKER.finditer(text or ""):
        tok = m.group(1).strip()
        if tok and tok not in seen:
            seen.append(tok)
    return seen


def check_citations(final_answer: str, passages: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """passages: dicts with evidence_id, doc_title (and optionally source_text/snippet).

    Returns {markers, resolved: {marker: [evidence_id, ...]}, unresolved: [marker, ...], n_markers}.
    """
    by_title: Dict[str, List[str]] = {}
    for p in passages:
        title = _norm(p.get("doc_title") or "")
        eid = p.get("evidence_id")
        if title and eid:
            bucket = by_title.setdefault(title, [])
            if eid not in bucket:
                bucket.append(eid)
    markers = extract_markers(final_answer)
    resolved: Dict[str, List[str]] = {}
    unresolved: List[str] = []
    for mk in markers:
        ids = by_title.get(_norm(mk))
        if ids:
            resolved[mk] = ids
        else:
            unresolved.append(mk)
    return {"markers": markers, "n_markers": len(markers), "resolved": resolved, "unresolved": unresolved}


def strip_unresolved_markers(text: str, passages: Iterable[Dict[str, Any]]):
    """Remove bracketed markers that match no retrieved passage title (placeholders, status tags, invented titles).

    Returns (cleaned_text, removed_markers). Markers that DO resolve are left untouched, so a valid inline
    citation is never altered; the caller should record `removed_markers` as a limitation, not hide it.
    """
    res = check_citations(text, passages)
    cleaned = text
    for mk in res["unresolved"]:
        cleaned = cleaned.replace(f"[{mk}]", "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\(\s*Source:\s*,?\s*", "(Source: ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned).strip()
    return cleaned, list(res["unresolved"])


_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("the and was were that this with from have has had are for not but his her their they them its who which into than then also been being about over under".split())


def _content_tokens(s: str) -> set:
    return {t for t in _WORD.findall(html.unescape(s or "").lower()) if len(t) > 2 and t not in _STOP}


def claim_evidence_map(claims: Iterable[Dict[str, Any]], passages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic claim -> passage mapping by content-word overlap (NOT produced by any LLM).

    passages need `evidence_id`, `doc_title` and full `text`. overlap = share of the claim's content words
    that occur in the best passage; it is a transparency aid for inspecting citations, not a verdict.
    """
    plist = [(p, _content_tokens(p.get("text") or p.get("source_text") or "")) for p in passages]
    out = []
    for c in claims:
        ct = _content_tokens(c.get("text") or "")
        best, best_ov = None, 0.0
        for p, pt in plist:
            ov = len(ct & pt) / len(ct) if ct else 0.0
            if ov > best_ov:
                best, best_ov = p, ov
        out.append({
            "claim_id": c.get("claim_id"), "state": c.get("state"), "claim": c.get("text"),
            "best_evidence_id": best["evidence_id"] if best else None,
            "best_doc_title": best.get("doc_title") if best else None,
            "content_word_overlap": round(best_ov, 3),
        })
    return out


LIMITATION_NOTE = "Some claims could not be verified and are not included."  # fixed, non-factual disclosure line

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_BRACKETED = re.compile(r"\[[^\[\]\n]{1,200}\]|\(\s*Source:[^)]*\)")


def audit_answer_sentences(final_answer: str, verified_claims: Iterable[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Flag final-answer sentences that no VERIFIED claim backs (deterministic, report-only; never edits the answer).

    A sentence is backed when at least `threshold` of its content words occur in one verified claim. Bracketed
    citations are ignored; very short fragments (< 3 content words) are skipped. Unbacked sentences are the model
    asserting something outside the verified claims, e.g. facts re-read from raw passages or the text of a repair
    whose re-verification failed.
    """
    claims = [(c, _content_tokens(c)) for c in verified_claims]
    out: List[Dict[str, Any]] = []
    for raw in _SENT_SPLIT.split(final_answer or ""):
        sent = _BRACKETED.sub("", raw).strip()
        if sent == LIMITATION_NOTE:
            continue
        toks = _content_tokens(sent)
        if len(toks) < 3:
            continue
        best_ov, best_claim = 0.0, None
        for c, ct in claims:
            ov = len(toks & ct) / len(toks)
            if ov > best_ov:
                best_ov, best_claim = ov, c
        out.append({"sentence": sent, "best_verified_claim_overlap": round(best_ov, 3),
                    "best_verified_claim": best_claim, "backed": best_ov >= threshold})
    return out
