"""
Unit tests for src/mir/search/reranker.py
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mir.search.faiss_index import SearchResult
from src.mir.search.reranker import (
    DiversityMMR, IdentityReranker, MetadataBoostReranker
)


def _make_results(n: int = 5, base_score: float = 0.9) -> list[SearchResult]:
    return [
        SearchResult(
            track_id=f"trk_{i}",
            score=base_score - i * 0.05,
            distance=i * 0.1,
            rank=i,
        )
        for i in range(n)
    ]


def _random_unit_emb(dim: int = 64) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class TestIdentityReranker:

    def test_preserves_order(self):
        results = _make_results(5)
        reranker = IdentityReranker()
        ranked = reranker.rerank(results)
        assert [r.track_id for r in ranked] == [f"trk_{i}" for i in range(5)]

    def test_preserves_scores(self):
        results = _make_results(5)
        reranker = IdentityReranker()
        ranked = reranker.rerank(results)
        for orig, re in zip(results, ranked):
            assert re.original_score == orig.score
            assert re.final_score == orig.score


# ---------------------------------------------------------------------------
# DiversityMMR
# ---------------------------------------------------------------------------

class TestDiversityMMR:

    def _build_reranker(self, embeddings: dict) -> DiversityMMR:
        def emb_fn(tid):
            return embeddings[tid]
        return DiversityMMR(lambda_param=0.5, embeddings_fn=emb_fn)

    def test_returns_same_count(self):
        results = _make_results(5)
        embs = {f"trk_{i}": _random_unit_emb() for i in range(5)}
        reranker = self._build_reranker(embs)
        query = _random_unit_emb()
        ranked = reranker.rerank(results, query_embedding=query)
        assert len(ranked) == 5

    def test_all_track_ids_present(self):
        results = _make_results(5)
        embs = {f"trk_{i}": _random_unit_emb() for i in range(5)}
        reranker = self._build_reranker(embs)
        ranked = reranker.rerank(results, query_embedding=_random_unit_emb())
        ids = {r.track_id for r in ranked}
        assert ids == {f"trk_{i}" for i in range(5)}

    def test_lambda1_similar_to_identity(self):
        """With lambda=1 (pure relevance), order should match original scores."""
        results = _make_results(5)
        embs = {f"trk_{i}": _random_unit_emb() for i in range(5)}
        reranker = DiversityMMR(lambda_param=1.0, embeddings_fn=lambda t: embs[t])
        ranked = reranker.rerank(results, query_embedding=_random_unit_emb())
        assert ranked[0].track_id == "trk_0"   # highest original score first

    def test_falls_back_without_embeddings(self):
        results = _make_results(3)
        reranker = DiversityMMR(lambda_param=0.5, embeddings_fn=None)
        ranked = reranker.rerank(results, query_embedding=_random_unit_emb())
        assert len(ranked) == 3

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            DiversityMMR(lambda_param=1.5)


# ---------------------------------------------------------------------------
# MetadataBoostReranker
# ---------------------------------------------------------------------------

class TestMetadataBoostReranker:

    def test_boost_promotes_track(self):
        results = _make_results(5)  # trk_0 has highest score
        # Boost trk_4 (lowest score) if genre == 'jazz'
        metadata = {f"trk_{i}": {"genre": "jazz" if i == 4 else "rock"} for i in range(5)}
        reranker = MetadataBoostReranker([("genre", "jazz", 100.0)])
        ranked = reranker.rerank(results, metadata=metadata)
        assert ranked[0].track_id == "trk_4"

    def test_penalty_demotes_track(self):
        results = _make_results(3)
        metadata = {"trk_0": {"explicit": True}, "trk_1": {}, "trk_2": {}}
        reranker = MetadataBoostReranker([("explicit", True, 0.01)])
        ranked = reranker.rerank(results, metadata=metadata)
        # trk_0 should drop below others
        assert ranked[0].track_id != "trk_0"

    def test_empty_metadata_safe(self):
        results = _make_results(3)
        reranker = MetadataBoostReranker([("genre", "pop", 1.5)])
        ranked = reranker.rerank(results, metadata={})
        assert len(ranked) == 3

    def test_no_matching_rules_preserves_order(self):
        results = _make_results(4)
        metadata = {f"trk_{i}": {"genre": "rock"} for i in range(4)}
        reranker = MetadataBoostReranker([("genre", "jazz", 2.0)])  # no match
        ranked = reranker.rerank(results, metadata=metadata)
        assert [r.track_id for r in ranked] == [f"trk_{i}" for i in range(4)]
