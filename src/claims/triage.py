"""Claim Triage Score (CTS) module.

Computes a transparent, configurable priority score for claims to determine
verification scheduling order under budget constraints.

CRITICAL METHODOLOGICAL NOTES:
1. CTS is a heuristic prioritization score in [0.0, 1.0], NOT a calibrated probability
   of falsity or hallucination.
2. CTS weights are initial working defaults for ranking, not learned bandit parameters
   or empirically optimal weights.
3. High CTS indicates high verification priority (due to high specificity, critical type,
   or low grounding in initial context).
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .models import ClaimRecord, ClaimType


# Basic stopword list for lexical overlap calculation
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "this", "that",
    "these", "those", "it", "its", "they", "their", "them", "which", "who", "whom",
    "what", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "should", "now"
}


@dataclass
class CTSConfig:
    """Configurable weights and feature coefficients for Claim Triage Scoring."""
    # Linear component weights (sum to 1.0)
    weight_claim_type: float = 0.30
    weight_specificity: float = 0.30
    weight_novelty: float = 0.25
    weight_complexity: float = 0.15

    # Type priority mapping (scale 0.0 - 1.0, higher = higher verification need)
    type_weights: Dict[ClaimType, float] = field(default_factory=lambda: {
        ClaimType.TEMPORAL_COMPARISON: 0.90,      # Temporal orderings are high-risk multi-hop failure points
        ClaimType.NUMERICAL_QUANTITATIVE: 0.85,  # Concrete metrics/quantities are highly falsifiable
        ClaimType.FACTUAL_RELATION: 0.75,        # Core relational facts
        ClaimType.EXISTENCE_NEGATION: 0.65,      # Negative assertions
        ClaimType.ATTRIBUTION: 0.50,             # Attribution meta-facts
        ClaimType.UNKNOWN_GENERAL: 0.40,         # General declarative propositions
        ClaimType.LIMITATION_UNCERTAINTY: 0.10,  # Explicit limitation statements (low verification priority)
        ClaimType.REFUSAL_UNANSWERABLE: 0.05,    # Refusal statements (lowest verification priority)
    })

    def __post_init__(self) -> None:
        total = (
            self.weight_claim_type +
            self.weight_specificity +
            self.weight_novelty +
            self.weight_complexity
        )
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"CTS component weights must sum to 1.0 (got {total:.4f}).")


class ClaimTriageScorer:
    """Computes Claim Triage Scores based on structural, contextual, and lexical features."""

    def __init__(self, config: Optional[CTSConfig] = None) -> None:
        self.config = config or CTSConfig()

    def _tokenize_content(self, text: str) -> Set[str]:
        """Extract lowercase content words (excluding punctuation and stopwords)."""
        if not text:
            return set()
        clean = text.lower()
        for ch in string.punctuation:
            clean = clean.replace(ch, " ")
        tokens = [w for w in clean.split() if len(w) > 1 and w not in STOPWORDS]
        return set(tokens)

    def extract_specificity(self, text: str) -> float:
        """Compute specificity feature in [0.0, 1.0].

        High specificity = presence of years, numerical figures, and capitalized proper nouns.
        """
        has_year = bool(re.search(r"\b(?:1[7-9][0-9]{2}|20[0-2][0-9])\b", text))
        has_number = bool(re.search(r"\b[0-9]+(?:\.[0-9]+)?\b", text))

        # Count capitalized words (excluding first word of sentence)
        words = text.split()
        capitalized = [w for w in words[1:] if w and w[0].isupper() and w.isalpha()]
        num_caps = len(capitalized)

        score = (
            0.40 * (1.0 if has_year else 0.0) +
            0.30 * (1.0 if has_number else 0.0) +
            0.30 * min(1.0, num_caps / 2.0)
        )
        return min(1.0, max(0.0, score))

    def extract_complexity(self, text: str) -> float:
        """Compute complexity feature in [0.0, 1.0].

        High complexity = longer clause length and higher density of named references.
        """
        words = text.split()
        token_count = len(words)
        caps_count = len([w for w in words if w and w[0].isupper()])

        score = (
            0.50 * min(1.0, token_count / 20.0) +
            0.50 * min(1.0, caps_count / 4.0)
        )
        return min(1.0, max(0.0, score))

    def extract_novelty(self, claim_text: str, context_text: Optional[str]) -> float:
        """Compute context novelty feature in [0.0, 1.0].

        Definition: Novelty = 1.0 - (overlap / claim_tokens).
        High novelty = low overlap with initial context (potential hallucination / ungrounded claim).
        If context is None or empty, novelty defaults to 1.0 (unsupported).
        """
        if not context_text or not context_text.strip():
            return 1.0

        claim_tokens = self._tokenize_content(claim_text)
        if not claim_tokens:
            return 0.5  # Neutral fallback for tokenless input

        context_tokens = self._tokenize_content(context_text)
        overlap = len(claim_tokens & context_tokens)
        overlap_ratio = overlap / len(claim_tokens)

        novelty = 1.0 - overlap_ratio
        return min(1.0, max(0.0, novelty))

    def compute_features(self, claim: ClaimRecord, context_text: Optional[str] = None) -> Dict[str, float]:
        """Extract all 4 CTS constituent features for a claim."""
        text = claim.normalized_text if claim.normalized_text else claim.original_text

        type_wt = self.config.type_weights.get(claim.claim_type, 0.40)
        spec = self.extract_specificity(text)
        comp = self.extract_complexity(text)
        nov = self.extract_novelty(text, context_text)

        return {
            "type_weight": round(type_wt, 4),
            "specificity": round(spec, 4),
            "novelty": round(nov, 4),
            "complexity": round(comp, 4),
        }

    def score_claim(self, claim: ClaimRecord, context_text: Optional[str] = None) -> float:
        """Compute and attach the Claim Triage Score (CTS) to a ClaimRecord."""
        if not getattr(claim, "is_substantive", True):
            # Non-substantive claims (limitations, refusals) receive minimal priority
            claim.triage_score = 0.05
            claim.triage_features = {
                "type_weight": 0.05,
                "specificity": 0.0,
                "novelty": 0.0,
                "complexity": 0.0,
            }
            return 0.05

        features = self.compute_features(claim, context_text)

        score = (
            self.config.weight_claim_type * features["type_weight"] +
            self.config.weight_specificity * features["specificity"] +
            self.config.weight_novelty * features["novelty"] +
            self.config.weight_complexity * features["complexity"]
        )

        score = round(min(1.0, max(0.0, score)), 4)
        claim.triage_score = score
        claim.triage_features = features
        return score

    def score_claims(self, claims: List[ClaimRecord], context_text: Optional[str] = None) -> List[ClaimRecord]:
        """Compute CTS for a collection of claims and return them sorted by descending priority."""
        for c in claims:
            self.score_claim(c, context_text)
        return sorted(claims, key=lambda c: c.triage_score if c.triage_score is not None else 0.0, reverse=True)
