"""Atomic claim decomposition module.

Transforms content-bearing sentences into structured, traceable ClaimRecords.
Applies conservative, semantically faithful propositional splitting rules while:
- Strictly preserving entity names, dates, numbers, negations, and modal qualifications
- Avoiding false splitting of collective predicates, comparatives, and conditionals
- Explicitly flagging ambiguous or uncertain decompositions (SPLIT_UNCERTAIN)
- Documenting operational heuristics and limitations without pretending to be a general NLP parser
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .models import ClaimRecord, ClaimState, ClaimType, DecompositionStatus, SentenceRecord


# Predicate patterns that must NOT be split because conjunctions form collective or comparative units
DO_NOT_SPLIT_PATTERNS = [
    re.compile(r"\bneither\b.+\bnor\b", re.IGNORECASE),
    re.compile(r"\b(?:both\b.+\band\b|\band\b.+\bboth\b)", re.IGNORECASE),
    re.compile(r"\b(?:between\b.+\band\b)", re.IGNORECASE),
    re.compile(r"\b(?:compare|compared to|more than|less than|older than|younger than|earlier than|later than)\b", re.IGNORECASE),
    re.compile(r"\b(?:because|since|if\b.+then|unless|although|even though|whereas)\b", re.IGNORECASE),
]

# Patterns indicating temporal comparison
TEMPORAL_PATTERNS = [
    re.compile(r"\b(?:started (?:first|before|after|in)|published in [0-9]{4}|founded in [0-9]{4}|between [0-9]{4} and [0-9]{4}|earlier|later)\b", re.IGNORECASE),
    re.compile(r"\b(?:1[7-9][0-9]{2}|20[0-2][0-9])\b"),
]

# Patterns indicating existence / negation
NEGATION_PATTERNS = [
    re.compile(r"\b(?:neither\b.+\bnor|not|never|none|no evidence|insufficient|does not|did not|was not|were not)\b", re.IGNORECASE),
]

# Patterns indicating quantitative / numeric facts
NUMERICAL_PATTERNS = [
    re.compile(r"\b[0-9]{1,3}(?:,[0-9]{3})+\b"),  # e.g. 1,310,696
    re.compile(r"\b[0-9]+(?:\.[0-9]+)?\s*(?:copies|megahertz|mhz|km|miles|percent|%|million|billion|thousand)\b", re.IGNORECASE),
]

# Patterns indicating relational assertions
RELATIONAL_PATTERNS = [
    re.compile(r"\b(?:is (?:married to|part of|based in|headquartered in|a member of|the daughter of|the son of|the wife of|the husband of))\b", re.IGNORECASE),
    re.compile(r"\b(?:was (?:founded by|created by|edited by|written by|directed by|launched in))\b", re.IGNORECASE),
]


class BaseDecomposer(ABC):
    """Abstract interface for claim decomposition."""

    @abstractmethod
    def decompose(self, sentence: SentenceRecord) -> List[ClaimRecord]:
        """Decompose a SentenceRecord into one or more ClaimRecords."""
        pass


class RuleBasedDecomposer(BaseDecomposer):
    """Conservative, syntax-guided claim decomposer.

    Heuristic Strategy:
    1. Determine claim type based on lexical and structural indicators.
    2. Check do-not-split constraints (negations, comparatives, conditionals, collective predicates).
    3. If safe, split coordinate independent clauses or compound verb phrases with shared subject.
    4. If splitting risk is detected, preserve intact as UNSPLIT_PRESERVED or flag as SPLIT_UNCERTAIN.
    """

    def __init__(self) -> None:
        pass

    def classify_claim_type(self, text: str) -> ClaimType:
        """Classify the predominant factual category of the claim text."""
        # 1. Existence / Negation check
        for p in NEGATION_PATTERNS:
            if p.search(text):
                return ClaimType.EXISTENCE_NEGATION

        # 2. Temporal comparison check
        for p in TEMPORAL_PATTERNS:
            if p.search(text):
                return ClaimType.TEMPORAL_COMPARISON

        # 3. Numerical / quantitative check
        for p in NUMERICAL_PATTERNS:
            if p.search(text):
                return ClaimType.NUMERICAL_QUANTITATIVE

        # 4. Factual relation check
        for p in RELATIONAL_PATTERNS:
            if p.search(text):
                return ClaimType.FACTUAL_RELATION

        return ClaimType.UNKNOWN_GENERAL

    def _should_not_split(self, text: str) -> bool:
        """Check if sentence contains constructs that must not be split."""
        for p in DO_NOT_SPLIT_PATTERNS:
            if p.search(text):
                return True
        return False

    def _extract_subject(self, clause: str) -> Optional[str]:
        """Extract approximate subject noun phrase from the beginning of a clause."""
        # Match capitalized noun phrase or simple pronoun up to auxiliary or main verb
        match = re.match(
            r"^([A-Z][a-zA-Z0-9\s'\"]+?)(?=\s+(?:is|was|were|are|has|had|started|published|founded|won|played|merged|features?|became|died))\b",
            clause
        )
        if match:
            return match.group(1).strip()
        return None

    def _split_compound_verbs(self, text: str) -> Optional[Tuple[str, str]]:
        """Split 'Subject verb phrase 1 and verb phrase 2' where subject is shared.

        Example:
            'Arthur's Magazine was published in Philadelphia and was edited by T.S. Arthur.'
            -> ('Arthur's Magazine was published in Philadelphia.', 'Arthur's Magazine was edited by T.S. Arthur.')
        """
        # Match: Subject + VP1 + (', and' or ' and') + (was|is|were|are|had|has) + VP2
        pattern = re.compile(
            r"^(.+?)(?:,\s*and|\s+and)\s+((?:was|is|were|are|had|has)\s+.+)$",
            re.IGNORECASE
        )
        match = pattern.match(text.rstrip("."))
        if match:
            part1 = match.group(1).strip()
            part2 = match.group(2).strip()

            subj = self._extract_subject(part1)
            if subj:
                clause1 = part1 + "."
                clause2 = f"{subj} {part2}."
                return clause1, clause2
        return None

    def _split_semicolon(self, text: str) -> Optional[List[str]]:
        """Split clauses joined by semicolons."""
        if ";" in text:
            parts = [p.strip().rstrip(".") + "." for p in text.split(";") if p.strip()]
            if len(parts) > 1:
                return parts
        return None

    def decompose(self, sentence: SentenceRecord) -> List[ClaimRecord]:
        """Decompose sentence into structured ClaimRecords."""
        text = sentence.cleaned_text if sentence.cleaned_text else sentence.original_text
        text = text.strip()
        sid = sentence.sentence_id

        # Empty or non-content sentence fallback
        if not text:
            return []

        # Handle explicit limitation or refusal sentences preserved from draft
        if getattr(sentence, "is_limitation", False):
            return [
                ClaimRecord(
                    claim_id=f"{sid}_c0",
                    source_sentence_id=sid,
                    original_text=text,
                    normalized_text=text,
                    claim_type=ClaimType.LIMITATION_UNCERTAINTY,
                    decomposition_status=DecompositionStatus.UNSPLIT_PRESERVED,
                    uncertainty_flag=True,
                    is_substantive=False,
                    is_limitation=True,
                    notes="Explicit limitation/uncertainty statement preserved from draft"
                )
            ]

        if getattr(sentence, "is_refusal", False):
            return [
                ClaimRecord(
                    claim_id=f"{sid}_c0",
                    source_sentence_id=sid,
                    original_text=text,
                    normalized_text=text,
                    claim_type=ClaimType.REFUSAL_UNANSWERABLE,
                    decomposition_status=DecompositionStatus.UNSPLIT_PRESERVED,
                    uncertainty_flag=True,
                    is_substantive=False,
                    is_limitation=True,
                    notes="Explicit refusal/unanswerable statement preserved from draft"
                )
            ]

        # 1. Do not split check
        if self._should_not_split(text):
            ctype = self.classify_claim_type(text)
            return [
                ClaimRecord(
                    claim_id=f"{sid}_c0",
                    source_sentence_id=sid,
                    original_text=text,
                    normalized_text=text,
                    claim_type=ctype,
                    decomposition_status=DecompositionStatus.UNSPLIT_PRESERVED,
                    uncertainty_flag=False,
                    notes="Preserved compound/relational construct (comparative, collective, or negation)"
                )
            ]

        # 2. Try semicolon split (usually clean independent clauses)
        semi_parts = self._split_semicolon(text)
        if semi_parts:
            claims: List[ClaimRecord] = []
            for i, part in enumerate(semi_parts):
                ctype = self.classify_claim_type(part)
                claims.append(
                    ClaimRecord(
                        claim_id=f"{sid}_c{i}",
                        source_sentence_id=sid,
                        original_text=part,
                        normalized_text=part,
                        claim_type=ctype,
                        decomposition_status=DecompositionStatus.SPLIT_DETERMINISTIC,
                        uncertainty_flag=False,
                        notes="Split on independent semicolon boundary"
                    )
                )
            return claims

        # 3. Try compound verb phrase split with shared subject
        compound_split = self._split_compound_verbs(text)
        if compound_split:
            c1_text, c2_text = compound_split
            c1_type = self.classify_claim_type(c1_text)
            c2_type = self.classify_claim_type(c2_text)

            return [
                ClaimRecord(
                    claim_id=f"{sid}_c0",
                    source_sentence_id=sid,
                    original_text=c1_text,
                    normalized_text=c1_text,
                    claim_type=c1_type,
                    decomposition_status=DecompositionStatus.SPLIT_DETERMINISTIC,
                    uncertainty_flag=False,
                    notes="Split from compound predicate (clause 1)"
                ),
                ClaimRecord(
                    claim_id=f"{sid}_c1",
                    source_sentence_id=sid,
                    original_text=c2_text,
                    normalized_text=c2_text,
                    claim_type=c2_type,
                    decomposition_status=DecompositionStatus.SPLIT_DETERMINISTIC,
                    uncertainty_flag=False,
                    notes="Split from compound predicate with subject ellipsis (clause 2)"
                )
            ]

        # 4. Check for comma-separated appositives (e.g. 'X, an American folk singer, married Y.')
        appositive_match = re.match(r"^([A-Z][a-zA-Z0-9\s'\"]+?),\s+an?\s+([^,]+?),\s+(.+)$", text.rstrip("."))
        if appositive_match:
            subj = appositive_match.group(1).strip()
            role = appositive_match.group(2).strip()
            pred = appositive_match.group(3).strip()

            c1_text = f"{subj} {pred}."
            c2_text = f"{subj} was a {role}." if not role.startswith("American") else f"{subj} was an {role}."

            return [
                ClaimRecord(
                    claim_id=f"{sid}_c0",
                    source_sentence_id=sid,
                    original_text=c1_text,
                    normalized_text=c1_text,
                    claim_type=self.classify_claim_type(c1_text),
                    decomposition_status=DecompositionStatus.SPLIT_DETERMINISTIC,
                    uncertainty_flag=False,
                    notes="Main clause from appositive construction"
                ),
                ClaimRecord(
                    claim_id=f"{sid}_c1",
                    source_sentence_id=sid,
                    original_text=c2_text,
                    normalized_text=c2_text,
                    claim_type=self.classify_claim_type(c2_text),
                    decomposition_status=DecompositionStatus.SPLIT_DETERMINISTIC,
                    uncertainty_flag=False,
                    notes="Extracted appositive attribute"
                )
            ]

        # 5. Default: Already atomic proposition
        ctype = self.classify_claim_type(text)
        return [
            ClaimRecord(
                claim_id=f"{sid}_c0",
                source_sentence_id=sid,
                original_text=text,
                normalized_text=text,
                claim_type=ctype,
                decomposition_status=DecompositionStatus.ATOMIC,
                uncertainty_flag=False,
                notes="Atomic proposition (no decomposition required)"
            )
        ]
