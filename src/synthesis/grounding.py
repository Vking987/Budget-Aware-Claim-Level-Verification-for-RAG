"""Claim-faithful final-answer construction and enforceable validation.

The final answer may only restate VERIFIED claims. This module provides
  * build_items:     verified claims -> deduplicated items, each with the passage that literally supports it
  * stitch_answer:   deterministic answer built from those items (grounded by construction)
  * validate_answer: structural check used to accept or reject any other candidate answer (e.g. an LLM rewrite)

validate_answer is NOT a lexical-overlap score. A candidate passes only if
  1. every factual sentence is exactly a union of WHOLE verified claims (it contains all content words and numbers of
     each claim it uses, and no content word outside their union), so no new fact, recombination or paraphrase-by-guess;
  2. no word that only occurs in an unresolved / contradicted / unverified-repair claim appears;
  3. no sentence repeats;
  4. no verified claim is dropped;
  5. each line cites exactly the sources chosen for the claims on that line (no foreign or placeholder citation).
What it does not prove: that a verified claim is true (the assessor decides that), or that merged sentences are grammatical.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .citations import _STOP, _norm, extract_markers

_WORD = re.compile(r"[a-z0-9]+")
_FACT = re.compile(r"\b(?:[A-Z][A-Za-z'’-]{2,}|\d[\d,.]*)\b")
_NOT_FACTS = frozenset("however there what when where while although because since after before".split())
_MARKER = re.compile(r"\[[^\[\]\n]{1,200}\]")
_SENT = re.compile(r"(?<=[.!?])\s+")
_DETERMINERS = frozenset("the a an this that these those their his her its they it he she".split())
MIN_SUPPORT_OVERLAP = 0.5


def tokens(text: str) -> List[str]:
    return _WORD.findall(html.unescape(text or "").lower())


def claim_tokens(text: str) -> set:
    """Content words (len > 2, not stop words) plus every number: the unit validation compares."""
    return {t for t in tokens(text) if t.isdigit() or (len(t) > 2 and t not in _STOP)}


def key_facts(text: str) -> set:
    """Capitalised names in a claim (numbers are handled separately, see unsupported_facts)."""
    out = set()
    for m in _FACT.finditer(html.unescape(text or "")):
        w = m.group(0).lower().strip(".,'’")
        if w and not w.isdigit() and w not in _NOT_FACTS and w not in _STOP:
            out.add(w)
    return out


def unsupported_facts(text: str, passage_tokens: set) -> List[str]:
    """Facts of the claim a passage fails to contain: EVERY number must occur literally (a wrong number is the
    dangerous error); names need only a majority, because descriptive name modifiers ("Disney Resort") are not facts."""
    nums = {t for t in tokens(text) if t.isdigit()}
    names = key_facts(text)
    missing_nums = sorted(n for n in nums if n not in passage_tokens)
    missing_names = sorted(n for n in names if n not in passage_tokens)
    if names and len(missing_names) > len(names) / 2:
        return missing_nums + missing_names
    return missing_nums


def support_for_claim(text: str, passages: Sequence[Any], min_overlap: float = MIN_SUPPORT_OVERLAP) -> Dict[str, Any]:
    """Pick the passage that supports the claim: content-word overlap >= min_overlap AND no unsupported number/name
    (see unsupported_facts). This is a lexical necessary condition, not entailment; the assessor's verdict is the
    entailment evidence."""
    ctoks = claim_tokens(text)
    rows = []
    for p in passages:
        all_toks = set(tokens(p.source_text))
        ov = len(ctoks & claim_tokens(p.source_text)) / len(ctoks) if ctoks else 0.0
        rows.append((p, ov, unsupported_facts(text, all_toks)))
    ok = [(p, ov) for p, ov, miss in rows if not miss and ov >= min_overlap]
    if ok:
        best, ov = max(ok, key=lambda r: (r[1], r[0].retrieval_score or 0.0))
        ids = [p.evidence_id for p, _ in ok if p.doc_title == best.doc_title]
        return {"title": best.doc_title, "evidence_ids": ids, "overlap": round(ov, 3), "missing_facts": []}
    if rows:
        p, ov, miss = max(rows, key=lambda r: (r[1], r[0].retrieval_score or 0.0))
        return {"title": None, "evidence_ids": [], "overlap": round(ov, 3), "missing_facts": miss}
    return {"title": None, "evidence_ids": [], "overlap": 0.0, "missing_facts": []}


def build_items(claims: Iterable[Any], repairs_by_claim: Dict[str, Dict[str, Any]], evidence_store: Dict[str, List[Any]],
                verified_claim_ids: set, allowed_titles: Optional[set] = None) -> List[Dict[str, Any]]:
    """Verified (or repaired AND re-verified) claims in draft order -> deduplicated items with their supporting passage."""
    items: List[Dict[str, Any]] = []
    for c in claims:
        if c.claim_id not in verified_claim_ids:
            continue
        rep = repairs_by_claim.get(c.claim_id)
        text = ((rep or {}).get("proposed_revision") or "").strip() or (c.normalized_text or c.original_text or "").strip()
        if not text:
            continue
        cands = [p for p in evidence_store.get(c.claim_id, []) if allowed_titles is None or p.doc_title in allowed_titles]
        items.append({"claim_id": c.claim_id, "text": text, **support_for_claim(text, cands)})
    # exact / token-equal duplicates, then claims wholly contained in a longer verified claim
    seen, uniq = set(), []
    for it in items:
        key = " ".join(tokens(it["text"]))
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    toks = [claim_tokens(i["text"]) for i in uniq]
    keep = []
    for i, it in enumerate(uniq):
        subsumed = any(j != i and toks[i] and toks[i] < toks[j] or (j < i and toks[i] and toks[i] == toks[j])
                       for j in range(len(uniq)))
        if not subsumed:
            keep.append(it)
    return keep


def _subject_and_rest(body: str):
    """('James and Oliver Phelps', 'played Fred ...') for a sentence led by a proper-noun phrase, else None."""
    words = body.split()
    k = 0
    while k < len(words) and (words[k][:1].isupper() or words[k].lower() in ("and", "of")):
        k += 1
    while k > 0 and words[k - 1].lower() in ("and", "of"):
        k -= 1
    if k < 2 or k >= len(words) or not words[k][:1].islower() or words[0].lower() in _DETERMINERS:
        return None
    return " ".join(words[:k]), " ".join(words[k:])


def _and_join(parts: List[str]) -> str:
    return parts[0] if len(parts) == 1 else (" and ".join(parts) if len(parts) == 2 else ", ".join(parts[:-1]) + ", and " + parts[-1])


def stitch_answer(items: Sequence[Dict[str, Any]]) -> str:
    """Deterministic answer from verified items: consecutive items sharing a source form one line with one citation;
    same-subject claims in a line are joined into one sentence ("X did A and is B")."""
    lines, i = [], 0
    items = list(items)
    while i < len(items):
        j = i + 1
        while items[i].get("title") and j < len(items) and items[j].get("title") == items[i]["title"]:
            j += 1
        sentences: List[Dict[str, Any]] = []
        for it in items[i:j]:
            body = it["text"].strip().rstrip(".!? ")
            sr = _subject_and_rest(body)
            if sr and sentences and sentences[-1]["subject"] == sr[0]:
                sentences[-1]["rests"].append(sr[1])
            else:
                sentences.append({"subject": sr[0] if sr else None, "full": body, "rests": [sr[1]] if sr else []})
        parts = []
        for s in sentences:
            parts.append((f"{s['subject']} {_and_join(s['rests'])}" if len(s["rests"]) > 1 else s["full"]) + ".")
        title = items[i].get("title")
        line = " ".join(parts)
        if title:  # citation sits inside the last sentence, before its period: "... in Hawaii [Title]."
            line = f"{line[:-1]} [{title}]."
        lines.append(line)
        i = j
    return "\n".join(lines)


def validate_answer(answer: str, items: Sequence[Dict[str, Any]], excluded_texts: Iterable[str] = ()) -> List[str]:
    """Return violations (empty list = the answer passes). See module docstring for exactly what is enforced."""
    items = list(items)
    if not items:
        return []
    vtoks = [claim_tokens(i["text"]) for i in items]
    vocab = set().union(*vtoks)
    titles = [_norm(i["title"]) if i.get("title") else None for i in items]
    used = [False] * len(items)
    answer_toks: set = set()
    seen: set = set()
    out: List[str] = []
    for line in (ln for ln in (answer or "").splitlines() if ln.strip()):
        markers = {_norm(m) for m in extract_markers(line)}
        line_used: set = set()
        for sent in (s.strip() for s in _SENT.split(_MARKER.sub(" ", line).strip()) if s.strip()):
            st = claim_tokens(sent)
            if not st:
                continue
            answer_toks |= st
            norm = " ".join(tokens(sent))
            if norm in seen:
                out.append(f"duplicate sentence: {sent[:80]!r}")
            seen.add(norm)
            contained = [k for k, t in enumerate(vtoks) if t and t <= st]
            if not contained:
                out.append(f"sentence contains no whole verified claim: {sent[:100]!r}")
                continue
            line_used.update(contained)
            for k in contained:
                used[k] = True
            extra = st - set().union(*[vtoks[k] for k in contained])
            if extra:
                out.append(f"words outside the verified claims {sorted(extra)}: {sent[:80]!r}")
        required = {titles[k] for k in line_used if titles[k]}
        for m in sorted(markers - required):
            out.append(f"citation [{m}] does not support any claim on its line")
        for m in sorted(required - markers):
            out.append(f"missing citation [{m}] for a claim on its line")
    for k, flag in enumerate(used):
        if not flag:
            out.append(f"verified claim dropped: {items[k]['text'][:90]!r}")
    for e in excluded_texts:
        leaked = sorted((claim_tokens(e) - vocab) & answer_toks)
        if leaked:
            out.append(f"reintroduces content of an unresolved/unverified claim: {leaked}")
    return out
