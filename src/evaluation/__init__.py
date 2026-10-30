"""Evaluation package for RAG research benchmark.
Contains standardized metrics for retrieval (Hit@K, Recall@K, Both@K)
and generation (Exact Match, Token F1).
"""

from .metrics import (
    normalize_answer,
    compute_exact_match,
    compute_token_f1,
    hit_at_k,
    recall_at_k,
    both_at_k,
    compute_retrieval_metrics,
    compute_generation_metrics,
)

__all__ = [
    "normalize_answer",
    "compute_exact_match",
    "compute_token_f1",
    "hit_at_k",
    "recall_at_k",
    "both_at_k",
    "compute_retrieval_metrics",
    "compute_generation_metrics",
]
