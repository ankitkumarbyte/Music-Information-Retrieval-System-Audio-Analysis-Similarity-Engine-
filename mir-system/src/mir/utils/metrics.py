"""
Evaluation Metrics
==================
Retrieval and classification metrics for MIR benchmarking.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def precision_at_k(relevant: list[bool], k: int) -> float:
    """Precision@K: fraction of top-K results that are relevant."""
    top_k = relevant[:k]
    return sum(top_k) / k if k > 0 else 0.0


def recall_at_k(relevant: list[bool], num_relevant: int, k: int) -> float:
    """Recall@K: fraction of all relevant items found in top-K."""
    if num_relevant == 0:
        return 0.0
    return sum(relevant[:k]) / num_relevant


def average_precision(relevant: list[bool]) -> float:
    """Mean of precision@K values at each relevant position."""
    hits = 0
    sum_prec = 0.0
    for i, rel in enumerate(relevant, 1):
        if rel:
            hits += 1
            sum_prec += hits / i
    total_relevant = sum(relevant)
    return sum_prec / total_relevant if total_relevant > 0 else 0.0


def mean_average_precision(all_relevant: list[list[bool]]) -> float:
    """MAP over a set of queries."""
    aps = [average_precision(rel) for rel in all_relevant]
    return float(np.mean(aps)) if aps else 0.0


def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.

    Parameters
    ----------
    relevance_scores : graded relevance scores (e.g. 0 or 1, or soft scores)
    k               : cut-off rank
    """
    k = min(k, len(relevance_scores))
    dcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(relevance_scores[:k])
    )
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def genre_classification_report(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
) -> dict:
    """Return accuracy, macro-F1, and per-class stats."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def instrument_detection_report(
    y_true: np.ndarray,   # (N, C) binary
    y_pred_proba: np.ndarray,  # (N, C) probabilities
    threshold: float = 0.5,
) -> dict:
    """Multi-label instrument detection metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "map": float(average_precision_score(y_true, y_pred_proba, average="macro")),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }
