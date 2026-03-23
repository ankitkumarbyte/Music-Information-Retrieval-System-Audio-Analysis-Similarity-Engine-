"""
Search Re-Ranker
=================
Post-retrieval re-ranking strategies applied on top of raw FAISS results.

Strategies
----------
- **Identity**      : pass-through (no re-ranking)
- **DiversityMMR**  : Maximal Marginal Relevance for diverse results
- **MetadataBoost** : boost / penalise results based on metadata fields
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.mir.search.faiss_index import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """Extended result after re-ranking."""

    track_id: str
    original_score: float      # raw cosine similarity from FAISS
    final_score: float         # score after re-ranking
    rank: int
    rerank_strategy: str


class BaseReranker(ABC):
    """Abstract re-ranker interface."""

    @abstractmethod
    def rerank(
        self,
        results: list[SearchResult],
        query_embedding: np.ndarray | None = None,
        metadata: dict[str, dict] | None = None,
    ) -> list[RankedResult]:
        ...


class IdentityReranker(BaseReranker):
    """No re-ranking — preserves FAISS order."""

    def rerank(self, results, query_embedding=None, metadata=None) -> list[RankedResult]:
        return [
            RankedResult(
                track_id=r.track_id,
                original_score=r.score,
                final_score=r.score,
                rank=i,
                rerank_strategy="identity",
            )
            for i, r in enumerate(results)
        ]


class DiversityMMR(BaseReranker):
    """
    Maximal Marginal Relevance re-ranking.

    Balances relevance (similarity to query) against diversity
    (dissimilarity to already-selected results).

    Parameters
    ----------
    lambda_param : float [0, 1]
        Weight for relevance vs diversity.
        λ=1.0 → pure relevance (same as identity).
        λ=0.5 → balanced.
        λ=0.0 → pure diversity.
    embeddings_fn : callable
        Function mapping track_id → np.ndarray embedding.
        Required for diversity computation.
    """

    def __init__(
        self,
        lambda_param: float = 0.7,
        embeddings_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be in [0, 1].")
        self.lambda_param = lambda_param
        self.embeddings_fn = embeddings_fn

    def rerank(
        self,
        results: list[SearchResult],
        query_embedding: np.ndarray | None = None,
        metadata: dict[str, dict] | None = None,
    ) -> list[RankedResult]:
        if not results:
            return []

        if self.embeddings_fn is None or query_embedding is None:
            logger.warning("MMR requires embeddings_fn and query_embedding. Falling back to identity.")
            return IdentityReranker().rerank(results)

        # Fetch embeddings for candidates
        candidates = []
        for r in results:
            try:
                emb = self.embeddings_fn(r.track_id)
                candidates.append((r, emb))
            except Exception:
                candidates.append((r, None))

        selected: list[tuple[SearchResult, np.ndarray]] = []
        remaining = candidates.copy()
        lam = self.lambda_param

        while remaining:
            best_idx = -1
            best_score = float("-inf")

            for i, (r, emb) in enumerate(remaining):
                relevance = r.score  # cosine similarity to query

                if emb is None or not selected:
                    diversity = 0.0
                else:
                    # Max similarity to already-selected items
                    sim_to_selected = max(
                        float(np.dot(emb, sel_emb) / (
                            np.linalg.norm(emb) * np.linalg.norm(sel_emb) + 1e-8
                        ))
                        for _, sel_emb in selected
                        if sel_emb is not None
                    )
                    diversity = 1.0 - sim_to_selected

                mmr_score = lam * relevance + (1 - lam) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            chosen_r, chosen_emb = remaining.pop(best_idx)
            selected.append((chosen_r, chosen_emb))

        return [
            RankedResult(
                track_id=r.track_id,
                original_score=r.score,
                final_score=r.score - i * 0.001,  # encode rank in score
                rank=i,
                rerank_strategy=f"mmr_lambda{lam}",
            )
            for i, (r, _) in enumerate(selected)
        ]


class MetadataBoostReranker(BaseReranker):
    """
    Apply multiplicative boost / penalty based on metadata fields.

    Parameters
    ----------
    boost_rules : list of (field, value, multiplier)
        Example: [("genre", "jazz", 1.3), ("year", "2020", 0.8)]
    """

    def __init__(self, boost_rules: list[tuple[str, object, float]]) -> None:
        self.boost_rules = boost_rules

    def rerank(
        self,
        results: list[SearchResult],
        query_embedding: np.ndarray | None = None,
        metadata: dict[str, dict] | None = None,
    ) -> list[RankedResult]:
        metadata = metadata or {}
        ranked = []
        for r in results:
            track_meta = metadata.get(r.track_id, {})
            boost = 1.0
            for field, value, multiplier in self.boost_rules:
                if track_meta.get(field) == value:
                    boost *= multiplier
            ranked.append((r, r.score * boost))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [
            RankedResult(
                track_id=r.track_id,
                original_score=r.score,
                final_score=boosted_score,
                rank=i,
                rerank_strategy="metadata_boost",
            )
            for i, (r, boosted_score) in enumerate(ranked)
        ]
