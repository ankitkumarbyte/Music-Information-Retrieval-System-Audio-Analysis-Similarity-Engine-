"""
Unit tests for src/mir/search/faiss_index.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.mir.search.faiss_index import FaissSearchEngine, SearchResult


DIM = 64   # smaller dim for test speed


def _random_embeddings(n: int, dim: int = DIM) -> np.ndarray:
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_engine(n_train: int = 500, nlist: int = 8) -> FaissSearchEngine:
    engine = FaissSearchEngine(dim=DIM, nlist=nlist, M=0, nprobe=4)
    train_data = _random_embeddings(n_train)
    engine.train(train_data)
    return engine


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTraining:

    def test_untrained_add_raises(self):
        engine = FaissSearchEngine(dim=DIM, nlist=8, M=0)
        with pytest.raises(RuntimeError, match="trained"):
            engine.add(_random_embeddings(5), ["a", "b", "c", "d", "e"])

    def test_trained_flag(self):
        engine = _make_engine()
        assert engine.stats.is_trained


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------

class TestAdd:

    def test_add_increases_count(self):
        engine = _make_engine()
        assert len(engine) == 0
        embeddings = _random_embeddings(10)
        engine.add(embeddings, [f"trk_{i}" for i in range(10)])
        assert len(engine) == 10

    def test_contains_after_add(self):
        engine = _make_engine()
        engine.add(_random_embeddings(3), ["a", "b", "c"])
        assert "a" in engine
        assert "z" not in engine

    def test_mismatched_lengths_raise(self):
        engine = _make_engine()
        with pytest.raises(ValueError):
            engine.add(_random_embeddings(3), ["a", "b"])


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:

    @pytest.fixture
    def populated_engine(self):
        engine = _make_engine()
        n = 100
        embeddings = _random_embeddings(n)
        track_ids = [f"trk_{i}" for i in range(n)]
        engine.add(embeddings, track_ids)
        return engine, embeddings, track_ids

    def test_search_returns_results(self, populated_engine):
        engine, embeddings, _ = populated_engine
        query = embeddings[0]
        results = engine.search(query, top_k=5)
        assert len(results) == 5

    def test_search_result_types(self, populated_engine):
        engine, embeddings, _ = populated_engine
        results = engine.search(embeddings[0], top_k=3)
        for r in results:
            assert isinstance(r, SearchResult)
            assert 0.0 <= r.score <= 1.0
            assert r.distance >= 0.0

    def test_search_on_empty_returns_empty(self):
        engine = _make_engine()
        results = engine.search(_random_embeddings(1).squeeze(), top_k=5)
        assert results == []

    def test_search_top_k_respected(self, populated_engine):
        engine, embeddings, _ = populated_engine
        for k in (1, 5, 10):
            results = engine.search(embeddings[0], top_k=k)
            assert len(results) <= k

    def test_self_similarity_highest(self, populated_engine):
        """Adding an exact copy and searching should put it first."""
        engine = _make_engine()
        vec = _random_embeddings(1)
        engine.add(vec, ["target"])
        # Add 50 random distractors
        engine.add(_random_embeddings(50), [f"d_{i}" for i in range(50)])
        results = engine.search(vec.squeeze(), top_k=1)
        assert results[0].track_id == "target"
        assert results[0].score > 0.99

    def test_exclude_ids(self, populated_engine):
        engine, embeddings, track_ids = populated_engine
        results = engine.search(embeddings[0], top_k=5, exclude_ids=[track_ids[0]])
        returned_ids = [r.track_id for r in results]
        assert track_ids[0] not in returned_ids


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------

class TestRemove:

    def test_remove_decreases_count(self):
        engine = _make_engine()
        engine.add(_random_embeddings(5), ["a", "b", "c", "d", "e"])
        engine.remove(["a", "b"])
        assert "a" not in engine
        assert "b" not in engine

    def test_remove_nonexistent_safe(self):
        engine = _make_engine()
        removed = engine.remove(["ghost"])
        assert removed == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_and_load(self):
        engine = _make_engine()
        vecs = _random_embeddings(20)
        ids = [f"p_{i}" for i in range(20)]
        engine.add(vecs, ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test.faiss"
            engine.save(idx_path)
            assert idx_path.exists()

            loaded = FaissSearchEngine.load(idx_path, dim=DIM, nlist=8, M=0)
            assert len(loaded) == 20
            for tid in ids:
                assert tid in loaded

    def test_search_consistency_after_load(self):
        engine = _make_engine()
        vecs = _random_embeddings(50)
        ids = [f"q_{i}" for i in range(50)]
        engine.add(vecs, ids)

        query = vecs[0]
        orig_results = engine.search(query, top_k=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "consistency.faiss"
            engine.save(idx_path)
            loaded = FaissSearchEngine.load(idx_path, dim=DIM, nlist=8, M=0)
            loaded_results = loaded.search(query, top_k=5)

        orig_ids = [r.track_id for r in orig_results]
        loaded_ids = [r.track_id for r in loaded_results]
        assert orig_ids == loaded_ids
