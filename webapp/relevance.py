"""Question analysis and relevance checks for retrieval and generation (rule-based; no model, no hard-coded answers).

Why this exists: sending a whole question ("Who created VIT University and when?") to a full-text search matches pages on words like "created" and
"when" (Balut, Hardbag) and on "VIT" as a prefix (Vitamin), while the right page (Vellore Institute of Technology) does not rank. These helpers
(1) pull the entity phrase out of the question, (2) build search queries around it, and (3) check that a retrieved page or passage actually
mentions the entity (an acronym such as VIT also matches a page title whose initials spell it).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

WH = {"who", "whom", "whose", "what", "when", "where", "which", "why", "how"}
AUX = {"is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "should", "has", "have", "had", "be", "been", "being", "am"}
IMPERATIVE = {"tell", "explain", "describe", "name", "give", "list", "define", "please", "summarize", "summarise"}
STOP = {"of", "a", "an", "the", "in", "on", "at", "to", "for", "and", "or", "by", "with", "as", "from", "that", "this", "it", "its", "into", "than", "then", "also", "not",
        "no", "there", "their", "his", "her", "they", "them", "he", "she", "you", "your", "we", "our", "i", "me", "my", "about", "over", "under", "between", "after",
        "before", "during", "while", "up", "down", "out", "off", "so", "such", "these", "those", "both", "each", "some", "any", "other", "more", "most", "very", "may",
        "might", "must", "shall", "if", "but", "are", "s", "t", "year", "years", "date", "time"}
CONNECTORS = {"of", "the", "and", "de", "la", "von", "van", "for", "in", "on"}
_WORD = re.compile(r"[\w'’-]+", re.UNICODE)


def stem(w: str) -> str:
    w = w.lower().strip("'’-")
    for suf in ("ingly", "edly", "ing", "ed", "es", "ly", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_stems(text: str) -> Set[str]:
    out = set()
    for w in _WORD.findall(text):
        lw = w.lower()
        if lw in WH or lw in AUX or lw in STOP or lw in IMPERATIVE or len(lw) < 2:
            continue
        out.add(stem(lw))
    return out


# Intent words a question uses vs the words encyclopedia text uses ("created" -> "founded"/"established"). Generic English, no topic content.
SYNONYM_GROUPS = [  # only words that carry the intent; very common words ("set", "form", "end", "led", "result") would match almost any passage
    {"creat", "found", "founder", "establish", "launch"},
    {"wrote", "write", "writ", "written", "author", "publish"},
    {"invent", "discover", "devis", "creat"},
    {"caus", "cause", "reason", "factor"},
    {"fall", "collapse", "decline", "overthrow"},
    {"die", "death", "kill", "murder"},
]
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def expand_intent(stems: Iterable[str]) -> Set[str]:
    out = set(stems)
    for g in SYNONYM_GROUPS:
        if out & g:
            out |= g
    return out


def asks_when(question: str) -> bool:
    ws = {w.lower() for w in _WORD.findall(question)}
    return "when" in ws or "year" in ws or "date" in ws


def has_year(text: str) -> bool:
    return bool(_YEAR.search(text))


def acronym(title: str) -> str:
    """Initials of the significant words of a title: 'Vellore Institute of Technology' -> 'vit'."""
    words = [w for w in _WORD.findall(title) if w.lower() not in CONNECTORS and w.lower() not in STOP]
    return "".join(w[0].lower() for w in words if w[0].isalpha())


@dataclass
class QuestionAnalysis:
    question: str
    entity_phrases: List[str] = field(default_factory=list)
    entity_tokens: List[str] = field(default_factory=list)  # stems
    content_tokens: List[str] = field(default_factory=list)  # base tokens plus their synonyms (matching vocabulary)
    base_tokens: List[str] = field(default_factory=list)  # stems of the question's own meaningful words, entity words included
    queries: List[str] = field(default_factory=list)  # search queries, best first

    @property
    def factoid(self) -> bool:
        """who / whom / whose / when / where / which questions ask for one fact; what / why / how questions ask for an explanation."""
        words = _WORD.findall(self.question.lower())
        return bool(words) and any(w in {"who", "whom", "whose", "when", "where", "which"} for w in words[:2])

    @property
    def intent_tokens(self) -> List[str]:
        """What the question asks about the entity (founded, published, caused...): base tokens that are not part of the entity."""
        return [t for t in self.base_tokens if t not in self.entity_tokens]

    def _hits(self, tokens: Sequence[str], have: Set[str]) -> int:
        return sum(1 for t in tokens if (expand_intent([t]) & have))

    def hit_fraction(self, title: str, text: str) -> float:
        """Share of the question's own words found in title+text; a synonym ('founded' for 'created') counts as a hit."""
        if not self.base_tokens:
            return 0.0
        return self._hits(self.base_tokens, content_stems(title) | content_stems(text)) / len(self.base_tokens)

    def intent_hits(self, text: str) -> int:
        return self._hits(self.intent_tokens, content_stems(text))

    def relevance(self, title: str, text: str) -> float:
        """Passage relevance for ranking: when the question asks something specific about the entity (founded, caused...), 70% of the score is
        whether the passage speaks to that and 30% whether it is about the entity; a passage that only names the entity cannot score high."""
        if not self.intent_tokens:
            return self.hit_fraction(title, text)
        ent = entity_coverage(self.entity_tokens, title, text) if self.entity_tokens else 1.0
        return 0.3 * ent + 0.7 * (self.intent_hits(text) / len(self.intent_tokens))

    def to_dict(self) -> Dict[str, object]:
        return {"entity_phrases": self.entity_phrases, "entity_tokens": self.entity_tokens, "intent_tokens": self.intent_tokens, "queries": self.queries}


def _is_cap(w: str) -> bool:
    return bool(w) and (w[0].isupper() or any(c.isdigit() for c in w)) and w.lower() not in WH and w.lower() not in AUX


def analyse_question(question: str) -> QuestionAnalysis:
    words = _WORD.findall(question)
    runs: List[List[str]] = []
    cur: List[str] = []
    for i, w in enumerate(words):
        lw = w.lower()
        starter = i == 0 and (lw in WH or lw in AUX or lw in IMPERATIVE or lw in STOP)
        if not starter and lw not in IMPERATIVE and _is_cap(w):
            cur.append(w)
        elif cur and lw in CONNECTORS and i + 1 < len(words) and _is_cap(words[i + 1]):
            cur.append(w)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    phrases = [" ".join(r) for r in runs]
    ent_tokens: List[str] = []
    for r in runs:
        for w in r:
            if w.lower() not in CONNECTORS and w.lower() not in STOP:
                s = stem(w)
                if s not in ent_tokens:
                    ent_tokens.append(s)
    base = sorted(content_stems(question))
    content = sorted(expand_intent(base))
    keywords = [w for w in words if w.lower() not in WH and w.lower() not in AUX and w.lower() not in STOP and w.lower() not in IMPERATIVE]
    queries: List[str] = []
    if phrases:
        queries.append(" ".join(phrases))
    kw = " ".join(keywords)
    if kw and kw not in queries:
        queries.append(kw)
    if not queries:
        queries.append(question.strip())
    return QuestionAnalysis(question=question, entity_phrases=phrases, entity_tokens=ent_tokens, content_tokens=content, base_tokens=base, queries=queries)


def required_entity_tokens(n: int) -> int:
    """How many entity tokens must be covered: all of up to two, two thirds of longer phrases."""
    return n if n <= 2 else math.ceil(0.67 * n)


def entity_coverage(entity_tokens: Sequence[str], title: str, text: str) -> float:
    """Fraction of the entity tokens found in the page title, the passage, or the title's acronym (VIT = Vellore Institute of Technology)."""
    if not entity_tokens:
        return 1.0
    have = content_stems(title) | content_stems(text)
    ac = acronym(title)
    hit = sum(1 for t in entity_tokens if t in have or t == ac)
    return hit / len(entity_tokens)


def title_match(entity_tokens: Sequence[str], title: str, lead: str = "", question_tokens: Iterable[str] = ()) -> float:
    """How exactly the page IS the entity: coverage of the entity tokens (title, lead, title acronym) times the share of the title's words the
    entity explains. 'Vellore Institute of Technology' (its acronym is VIT) scores 1.0 for 'VIT University'; 'VIT Bhopal University' only 2/3 because
    'Bhopal' is extra; 'Frankenstein' beats 'Frankenstein (1931 film)'. Title words that appear in the question (e.g. 'fall') are explained too, so
    'Fall of the Western Roman Empire' is an exact match for a question about that fall."""
    if not entity_tokens:
        return 1.0
    cov = entity_coverage(entity_tokens, title, lead)
    ts = content_stems(title)
    explained_by = set(entity_tokens) | set(question_tokens)
    if acronym(title) in entity_tokens:
        precision = 1.0
    else:
        precision = (sum(1 for t in ts if t in explained_by) / len(ts)) if ts else 0.0
    return cov * precision


def mentions_entity(entity_tokens: Sequence[str], title: str, text: str) -> bool:
    if not entity_tokens:
        return True
    return entity_coverage(entity_tokens, title, text) * len(entity_tokens) + 1e-9 >= required_entity_tokens(len(entity_tokens))


def question_overlap(analysis: QuestionAnalysis, title: str, text: str) -> float:
    return analysis.relevance(title, text)


def passages_support_question(analysis: QuestionAnalysis, passages: Sequence) -> Dict[str, object]:
    """Relevance gate over retrieved passages (any source). Entity questions need a passage that mentions the entity; questions with no
    recognisable entity need some word overlap with the question. Returns {ok, reason, kept}."""
    kept = []
    for p in passages:
        title, text = p.doc_title, p.source_text
        if analysis.entity_tokens:
            if mentions_entity(analysis.entity_tokens, title, text):
                kept.append(p)
        elif analysis.hit_fraction(title, text) > 0:
            kept.append(p)
    if kept:
        return {"ok": True, "reason": "", "kept": kept}
    what = ("the entity " + " / ".join(analysis.entity_phrases)) if analysis.entity_phrases else "the question's key terms"
    return {"ok": False, "kept": [], "reason": f"None of the {len(passages)} retrieved passages mentions {what}."}


def claim_is_on_topic(analysis: QuestionAnalysis, claim_text: str, is_first: bool = False) -> bool:
    """A claim stays when it speaks to what the question asks about the entity (a word like 'founded' or 'published', synonyms count). Sharing only
    the entity's name is not enough: 'VIT was ranked 801-1000' mentions VIT but does not answer who founded it. A question with no intent words
    ('What is Frankenstein?') accepts any claim that shares one of its words. The first claim, which opens with the direct answer, always stays."""
    if is_first:
        return True
    if analysis.intent_tokens and analysis.factoid:
        return analysis.intent_hits(claim_text) > 0  # who / when / where: the claim must say something about what was asked
    if analysis.intent_tokens:  # what / why / how: causal or descriptive detail often avoids the question's words, so naming the entity also counts
        return analysis.intent_hits(claim_text) > 0 or entity_coverage(analysis.entity_tokens, "", claim_text) > 0
    return analysis.hit_fraction("", claim_text) > 0
