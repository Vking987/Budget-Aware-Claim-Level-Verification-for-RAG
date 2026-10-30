"""Content-bearing sentence filtering module.

Identifies and removes conversational boilerplate and non-substantive preamble
while strictly preserving:
- Factual claims and propositions
- Epistemic uncertainty statements ("insufficient information", "unknown")
- Negations ("neither ... nor ...", "did not")
- Temporal indicators and numerical qualifications
- Full character traceability back to the source answer
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .models import SentenceRecord


# Leading conversational prefixes that should be stripped to reveal the core proposition
LEADING_FILLER_PATTERNS = [
    re.compile(r"^(?:based on (?:the )?(?:provided )?context,?\s*)", re.IGNORECASE),
    re.compile(r"^(?:according to (?:the )?(?:provided )?(?:documents?|context|text),?\s*)", re.IGNORECASE),
    re.compile(
        r"^(?:the information (?:provided )?in (?:the )?(?:documents?|context) (?:and [a-z0-9\s]+)?(?:indicates?|states?|shows?|suggests?|mentions?) that\s*)",
        re.IGNORECASE
    ),
    re.compile(r"^(?:from the (?:provided )?(?:context|documents?),?\s*)", re.IGNORECASE),
    re.compile(r"^(?:as stated in (?:the )?(?:documents?|context),?\s*)", re.IGNORECASE),
    re.compile(r"^(?:to answer (?:the|your) question,?\s*)", re.IGNORECASE),
    re.compile(r"^(?:sure,?(?:\s*here is (?:the )?answer:?)?\s*)", re.IGNORECASE),
    re.compile(r"^(?:here is the answer:?\s*)", re.IGNORECASE),
]

# Sentences that are purely conversational meta-commentary with zero factual content
PURE_FILLER_PATTERNS = [
    re.compile(r"^(?:sure|certainly|of course)[!.]?$", re.IGNORECASE),
    re.compile(r"^here is the answer[:.]?$", re.IGNORECASE),
    re.compile(r"^(?:i hope this helps|let me know if you need more information)[!.]?$", re.IGNORECASE),
    re.compile(r"^thank you[!.]?$", re.IGNORECASE),
    re.compile(r"^(?:hello|hi|hey)there[!.]?$", re.IGNORECASE),
]

# Sentences that represent explicit refusals or inability to answer
REFUSAL_PATTERNS = [
    re.compile(r"^(?:i cannot answer|i am unable to answer|i can't answer|cannot answer this question|unable to answer based on)\b", re.IGNORECASE),
    re.compile(r"^(?:sorry,? (?:but )?i cannot|i apologize,? (?:but )?i cannot)\b", re.IGNORECASE),
    re.compile(r"^(?:as an ai,? i (?:cannot|can't|am unable to))\b", re.IGNORECASE),
]

# Sentences that represent explicit epistemic limitations or question-premise uncertainty
LIMITATION_PATTERNS = [
    re.compile(r"^(?:the limitation is|a limitation is|limitations?:|note:)\s*", re.IGNORECASE),
    re.compile(r"\b(?:the question does not (?:specify|state|mention|provide|indicate))\b", re.IGNORECASE),
    re.compile(r"\b(?:cannot be answered (?:from|based on)|not enough context to determine|cannot be determined from the question)\b", re.IGNORECASE),
    re.compile(r"\b(?:the provided (?:context|passages|documents?) do(?:es)? not (?:contain|mention|state|provide))\b", re.IGNORECASE),
    re.compile(r"^(?:there is no information|no information is provided|there is not enough information to answer)\b", re.IGNORECASE),
    re.compile(r"^(?:it is unknown|it remains unclear|it is not specified)\b", re.IGNORECASE),
]

# Patterns that MUST be preserved as substantive factual content even if they contain qualifying terms
PRESERVE_SAFEGUARDS = [
    re.compile(r"\b(?:neither\b.+\bnor|did not|was not|were not|never|none|not|no)\b", re.IGNORECASE),
    re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b"),  # Years (1000-2099)
    re.compile(r"\b[0-9]+(?:\.[0-9]+)?\b"),       # Numbers
    re.compile(r"\b(?:except for|apart from|other than|with the exception of|unless)\b", re.IGNORECASE),
]


class ContentBearingFilter:
    """Filter that separates content-bearing propositions from conversational noise."""

    def __init__(self, fallback_to_preserve: bool = True) -> None:
        """Initialize filter.

        Args:
            fallback_to_preserve: When True, any ambiguous or borderline sentence
                                  is preserved rather than dropped.
        """
        self.fallback_to_preserve = fallback_to_preserve

    def split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences while tracking start and end character offsets.

        Handles common abbreviations (e.g., T.S., U.S., Dr., etc.) without false breaks.
        """
        if not text or not text.strip():
            return []

        # Mask common period-bearing abbreviations and initials temporarily for splitting
        # e.g., T.S. -> T@@S@@, Dr. -> Dr@@, etc.
        masked_text = re.sub(r"\b([A-Z])\.", r"\1@DOT@", text)
        masked_text = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.", r"\1@DOT@", masked_text, flags=re.IGNORECASE)

        sentence_spans: List[Tuple[str, int, int]] = []
        pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|(?<=\n)\s*")

        pos = 0
        raw_parts = [p.replace("@DOT@", ".") for p in pattern.split(masked_text) if p.strip()]
        for part in raw_parts:
            start = text.find(part, pos)
            if start == -1:
                start = pos
            end = start + len(part)
            sentence_spans.append((part.strip(), start, end))
            pos = end

        # Fallback if regex produced nothing
        if not sentence_spans and text.strip():
            sentence_spans = [(text.strip(), 0, len(text))]

        return sentence_spans

    def filter_sentence(self, sentence_text: str, char_start: int, char_end: int, index: int) -> SentenceRecord:
        """Analyze a single sentence, remove leading boilerplate, and classify content-bearing status."""
        sid = f"s_{index}"
        raw = sentence_text.strip()

        # 1. Check pure conversational filler patterns (e.g. "Sure!", "Here is the answer.")
        for p in PURE_FILLER_PATTERNS:
            if p.match(raw):
                return SentenceRecord(
                    sentence_id=sid,
                    original_text=raw,
                    cleaned_text="",
                    is_content_bearing=False,
                    filter_reason="pure_conversational_filler",
                    char_start=char_start,
                    char_end=char_end,
                    confidence=0.95,
                    is_substantive=False,
                    is_limitation=False,
                    is_refusal=False,
                )

        # 2. Check explicit refusals
        for p in REFUSAL_PATTERNS:
            if p.search(raw):
                return SentenceRecord(
                    sentence_id=sid,
                    original_text=raw,
                    cleaned_text=raw,
                    is_content_bearing=True,
                    filter_reason="explicit_refusal",
                    char_start=char_start,
                    char_end=char_end,
                    confidence=0.95,
                    is_substantive=False,
                    is_limitation=False,
                    is_refusal=True,
                )

        # 3. Check explicit limitation / uncertainty statements (e.g. "The limitation is: ...")
        for p in LIMITATION_PATTERNS:
            if p.search(raw):
                return SentenceRecord(
                    sentence_id=sid,
                    original_text=raw,
                    cleaned_text=raw,
                    is_content_bearing=True,
                    filter_reason="explicit_limitation_statement",
                    char_start=char_start,
                    char_end=char_end,
                    confidence=0.95,
                    is_substantive=False,
                    is_limitation=True,
                    is_refusal=False,
                )

        # 4. Check explicit preservation safeguards (negation, numerical, temporal, exceptions)
        for p in PRESERVE_SAFEGUARDS:
            if p.search(raw):
                # Still strip leading conversational boilerplate if present
                cleaned = raw
                for lp in LEADING_FILLER_PATTERNS:
                    cleaned = lp.sub("", cleaned).strip()
                if cleaned and cleaned[0].islower():
                    cleaned = cleaned[0].upper() + cleaned[1:]

                return SentenceRecord(
                    sentence_id=sid,
                    original_text=raw,
                    cleaned_text=cleaned if cleaned else raw,
                    is_content_bearing=True,
                    filter_reason="safeguard_preserved (negation/numeric/temporal)",
                    char_start=char_start,
                    char_end=char_end,
                    confidence=1.0,
                    is_substantive=True,
                    is_limitation=False,
                    is_refusal=False,
                )

        # 5. Strip leading conversational prefixes
        cleaned = raw
        was_stripped = False
        for lp in LEADING_FILLER_PATTERNS:
            new_cleaned = lp.sub("", cleaned).strip()
            if new_cleaned != cleaned:
                cleaned = new_cleaned
                was_stripped = True

        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]

        # If stripped to empty, it was pure filler
        if not cleaned:
            return SentenceRecord(
                sentence_id=sid,
                original_text=raw,
                cleaned_text="",
                is_content_bearing=False,
                filter_reason="stripped_to_empty_filler",
                char_start=char_start,
                char_end=char_end,
                confidence=0.9,
                is_substantive=False,
                is_limitation=False,
                is_refusal=False,
            )

        # Non-empty cleaned sentence -> Substantive content-bearing proposition
        reason = "content_bearing_prefix_stripped" if was_stripped else "content_bearing_declarative"
        return SentenceRecord(
            sentence_id=sid,
            original_text=raw,
            cleaned_text=cleaned,
            is_content_bearing=True,
            filter_reason=reason,
            char_start=char_start,
            char_end=char_end,
            confidence=0.9 if was_stripped else 1.0,
            is_substantive=True,
            is_limitation=False,
            is_refusal=False,
        )

    def process(self, answer_text: str) -> List[SentenceRecord]:
        """Process an answer string into a list of SentenceRecord objects."""
        spans = self.split_sentences(answer_text)
        records: List[SentenceRecord] = []
        for i, (span_text, start, end) in enumerate(spans):
            rec = self.filter_sentence(span_text, start, end, i)
            records.append(rec)
        return records
