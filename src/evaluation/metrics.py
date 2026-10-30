"""Standardized evaluation metrics for multi-hop retrieval and QA generation.

Defines:
- Document Retrieval Metrics:
  - Hit@K: Fraction of queries with at least one supporting document in top-K.
  - Recall@K: Fraction of gold supporting documents retrieved in top-K (|gold & top_k| / |gold|).
  - Both@K: Fraction of queries where all gold supporting documents are retrieved in top-K.
- Answer Generation Metrics:
  - Exact Match (EM): Binary indicator whether normalized prediction matches normalized ground truth.
  - Token F1: Macro token-level precision, recall, and harmonic mean F1.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Set


# ==============================================================================
# Text Normalization for Question Answering (Standard SQuAD / HotpotQA rules)
# ==============================================================================

def normalize_answer(text: str | None) -> str:
    """Normalize text by lowercasing, removing punctuation, articles, and extra whitespace.

    Args:
        text: Input string or None.

    Returns:
        Cleaned, normalized string.
    """
    if text is None:
        return ""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


# ==============================================================================
# Answer Generation Metrics
# ==============================================================================

def compute_exact_match(prediction: str | None, ground_truth: str | None) -> float:
    """Compute binary Exact Match (EM) score between prediction and reference.

    Returns 1.0 if normalized strings are identical, else 0.0.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_token_f1(prediction: str | None, ground_truth: str | None) -> float:
    """Compute token-level F1 score between prediction and reference.

    Tokenizes normalized strings by whitespace and computes token overlap.
    If both strings are empty after normalization, returns 1.0.
    If either is empty while the other is not, returns 0.0.
    """
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)

    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return float(f1)


def compute_generation_metrics(
    prediction: str | None,
    ground_truth: str | None
) -> Dict[str, float]:
    """Compute both Exact Match and Token F1 for a predicted answer."""
    return {
        "exact_match": compute_exact_match(prediction, ground_truth),
        "f1": compute_token_f1(prediction, ground_truth),
    }


# ==============================================================================
# Document Retrieval Metrics
# ==============================================================================

def hit_at_k(
    retrieved_titles: Sequence[str],
    gold_titles: Set[str] | Sequence[str],
    k: int
) -> float:
    """Compute Hit@K (at least one gold supporting document retrieved in top K).

    Args:
        retrieved_titles: Ordered sequence of retrieved document titles.
        gold_titles: Set or sequence of ground-truth document titles.
        k: Cutoff rank (1-indexed).

    Returns:
        1.0 if intersection of gold titles and top-K retrieved titles is non-empty,
        0.0 otherwise. If gold_titles is empty, returns 0.0.
    """
    golds = set(gold_titles)
    if not golds or k <= 0:
        return 0.0

    top_k_set = set(retrieved_titles[:k])
    return 1.0 if bool(golds & top_k_set) else 0.0


def recall_at_k(
    retrieved_titles: Sequence[str],
    gold_titles: Set[str] | Sequence[str],
    k: int
) -> float:
    """Compute true multi-hop Recall@K (|gold & retrieved[:k]| / |gold|).

    Unlike Hit@K, Recall@K measures the proportion of required supporting
    documents retrieved at cutoff K.

    Args:
        retrieved_titles: Ordered sequence of retrieved document titles.
        gold_titles: Set or sequence of ground-truth document titles.
        k: Cutoff rank (1-indexed).

    Returns:
        Float between 0.0 and 1.0. If gold_titles is empty, returns 0.0.
    """
    golds = set(gold_titles)
    if not golds or k <= 0:
        return 0.0

    top_k_set = set(retrieved_titles[:k])
    overlap = len(golds & top_k_set)
    return float(overlap / len(golds))


def both_at_k(
    retrieved_titles: Sequence[str],
    gold_titles: Set[str] | Sequence[str],
    k: int
) -> float:
    """Compute Both@K / Complete Set Recall (all gold documents retrieved in top K).

    Args:
        retrieved_titles: Ordered sequence of retrieved document titles.
        gold_titles: Set or sequence of ground-truth document titles.
        k: Cutoff rank (1-indexed).

    Returns:
        1.0 if all gold titles are present in top-K retrieved titles,
        0.0 otherwise. If gold_titles is empty, returns 0.0.
    """
    golds = set(gold_titles)
    if not golds or k <= 0:
        return 0.0

    top_k_set = set(retrieved_titles[:k])
    return 1.0 if golds.issubset(top_k_set) else 0.0


def compute_retrieval_metrics(
    retrieved_titles: Sequence[str],
    gold_titles: Set[str] | Sequence[str],
    cutoffs: Sequence[int] = (1, 3, 5, 10)
) -> Dict[str, float]:
    """Compute Hit@K, Recall@K, and Both@K across all requested cutoffs."""
    golds = set(gold_titles)
    metrics: Dict[str, float] = {}

    for k in cutoffs:
        metrics[f"hit_{k}"] = hit_at_k(retrieved_titles, golds, k)
        metrics[f"recall_{k}"] = recall_at_k(retrieved_titles, golds, k)
        metrics[f"both_{k}"] = both_at_k(retrieved_titles, golds, k)

    return metrics
