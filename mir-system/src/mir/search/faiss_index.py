"""
FAISS Similarity Search Engine
===============================
Manages an IVF+PQ FAISS index for approximate nearest-neighbour
search over 256-dimensional audio embeddings.

Features
--------
- IVFFlat / IVFPQ index with configurable compression
- Thread-safe read/write with RWLock
- Incremental add without full rebuild
- Persistent snapshot to disk
- Cosine distance via L2 on pre-normalised vectors
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single similarity search result."""

    track_id: str
    score: float        # cosine similarity [0, 1]
    distance: float     # L2 distance (lower = more similar)
    rank: int


@dataclass
class IndexStats:
    """Runtime statistics for the FAISS index."""

    num_vectors: int = 0
    embedding_dim: int = 256
    index_type: str = ""
    is_trained: bool = False
    index_size_bytes: int = 0
    nlist: int = 0
    nprobe: int = 0


class FaissSearchEngine:
    """
    Thread-safe FAISS-based ANN search engine.

    Parameters
    ----------
    dim : int
        Embedding dimension (must match encoder output).
    nlist : int
        Number of IVF Voronoi cells.
    M : int
        Number of PQ sub-quantizers (set to 0 to disable PQ).
    nbits : int
        Bits per sub-quantizer (used when M > 0).
    nprobe : int
        Number of cells to inspect during search.
    use_gpu : bool
        Move index to GPU if available.

    Example
    -------
    >>> engine = FaissSearchEngine(dim=256)
    >>> engine.add(embeddings, track_ids)
    >>> results = engine.search(query_embedding, top_k=10)
    """

    def __init__(
        self,
        dim: int = 256,
        nlist: int = 256,
        M: int = 16,
        nbits: int = 8,
        nprobe: int = 32,
        use_gpu: bool = False,
    ) -> None:
        self.dim = dim
        self.nlist = nlist
        self.M = M
        self.nbits = nbits
        self.nprobe = nprobe
        self.use_gpu = use_gpu

        self._index: faiss.Index = self._build_index()
        self._id_map: dict[int, str] = {}      # faiss int id → track_id string
        self._rev_map: dict[str, int] = {}     # track_id → faiss int id
        self._next_id: int = 0
        self._lock = threading.RLock()

        logger.info(
            "FaissSearchEngine | dim=%d nlist=%d M=%d nprobe=%d GPU=%s",
            dim, nlist, M, nprobe, use_gpu,
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> faiss.Index:
        """Build the base FAISS index."""
        quantizer = faiss.IndexFlatL2(self.dim)
        if self.M > 0:
            index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.M, self.nbits)
        else:
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)

        index.nprobe = self.nprobe

        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("FAISS index moved to GPU 0")

        return index

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, embeddings: np.ndarray) -> None:
        """
        Train the IVF quantizer.

        Must be called before the first ``add`` if the index has not been
        loaded from disk. Requires at least ``nlist * 39`` training vectors.

        Parameters
        ----------
        embeddings : np.ndarray  shape (N, dim)  dtype float32
        """
        embeddings = self._validate(embeddings)
        min_required = self.nlist * 39
        if len(embeddings) < min_required:
            logger.warning(
                "Training with %d vectors; recommended minimum is %d for nlist=%d.",
                len(embeddings), min_required, self.nlist,
            )
        with self._lock:
            logger.info("Training FAISS index on %d vectors ...", len(embeddings))
            t0 = time.perf_counter()
            self._index.train(embeddings)
            logger.info("Training complete in %.2fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, track_ids: list[str]) -> None:
        """
        Add embeddings to the index.

        Parameters
        ----------
        embeddings : np.ndarray  shape (N, dim)
        track_ids  : list[str]   unique track identifiers, len == N
        """
        if not self._index.is_trained:
            raise RuntimeError("Index must be trained before adding vectors. Call .train() first.")
        embeddings = self._validate(embeddings)
        if len(embeddings) != len(track_ids):
            raise ValueError("embeddings and track_ids must have the same length.")

        with self._lock:
            ids = np.arange(self._next_id, self._next_id + len(track_ids), dtype=np.int64)
            self._index.add_with_ids(embeddings, ids)
            for i, tid in enumerate(track_ids):
                faiss_id = int(ids[i])
                self._id_map[faiss_id] = tid
                self._rev_map[tid] = faiss_id
            self._next_id += len(track_ids)
        logger.debug("Added %d vectors. Index size: %d", len(track_ids), self._index.ntotal)

    def remove(self, track_ids: list[str]) -> int:
        """
        Remove tracks from the index.

        Returns
        -------
        int  Number of vectors actually removed.
        """
        with self._lock:
            ids_to_remove = []
            for tid in track_ids:
                if tid in self._rev_map:
                    faiss_id = self._rev_map.pop(tid)
                    del self._id_map[faiss_id]
                    ids_to_remove.append(faiss_id)

            if ids_to_remove:
                id_selector = faiss.IDSelectorBatch(
                    np.array(ids_to_remove, dtype=np.int64)
                )
                n_removed = self._index.remove_ids(id_selector)
                logger.info("Removed %d vectors from index", n_removed)
                return n_removed
        return 0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        exclude_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Find top-k nearest neighbours for a query embedding.

        Parameters
        ----------
        query : np.ndarray  shape (dim,) or (1, dim)
        top_k : int
        exclude_ids : list[str], optional
            Track IDs to exclude from results (e.g. the query track itself).

        Returns
        -------
        list[SearchResult]  ranked by similarity (highest first)
        """
        query = self._validate(query.reshape(1, -1))
        if self._index.ntotal == 0:
            return []

        fetch_k = top_k + len(exclude_ids or []) + 1  # over-fetch to handle exclusions
        with self._lock:
            distances, faiss_ids = self._index.search(query, min(fetch_k, self._index.ntotal))

        results = []
        for rank, (dist, fid) in enumerate(zip(distances[0], faiss_ids[0])):
            if fid == -1:
                continue
            tid = self._id_map.get(int(fid))
            if tid is None or (exclude_ids and tid in exclude_ids):
                continue
            # Convert L2 distance to cosine similarity (vectors are unit-normalised)
            similarity = max(0.0, 1.0 - float(dist) / 2.0)
            results.append(SearchResult(
                track_id=tid,
                score=similarity,
                distance=float(dist),
                rank=rank,
            ))
            if len(results) >= top_k:
                break

        return results

    def batch_search(
        self,
        queries: np.ndarray,
        top_k: int = 10,
    ) -> list[list[SearchResult]]:
        """Search for multiple queries in a single FAISS call."""
        queries = self._validate(queries)
        with self._lock:
            distances, faiss_ids = self._index.search(queries, top_k)

        all_results = []
        for row_dist, row_ids in zip(distances, faiss_ids):
            row = []
            for rank, (dist, fid) in enumerate(zip(row_dist, row_ids)):
                if fid == -1:
                    continue
                tid = self._id_map.get(int(fid))
                if tid is None:
                    continue
                similarity = max(0.0, 1.0 - float(dist) / 2.0)
                row.append(SearchResult(track_id=tid, score=similarity, distance=float(dist), rank=rank))
            all_results.append(row)
        return all_results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist index and ID map to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            cpu_index = faiss.index_gpu_to_cpu(self._index) if self.use_gpu else self._index
            faiss.write_index(cpu_index, str(path))
            np.save(str(path) + ".ids.npy", self._id_map)
        logger.info("Index saved to %s (%d vectors)", path, self._index.ntotal)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "FaissSearchEngine":
        """Load a persisted index from disk."""
        path = Path(path)
        engine = cls(**kwargs)
        engine._index = faiss.read_index(str(path))
        engine._index.nprobe = engine.nprobe
        id_map_path = str(path) + ".ids.npy"
        if os.path.exists(id_map_path):
            loaded = np.load(id_map_path, allow_pickle=True).item()
            engine._id_map = loaded
            engine._rev_map = {v: k for k, v in loaded.items()}
            engine._next_id = max(loaded.keys(), default=-1) + 1
        logger.info("Loaded index from %s (%d vectors)", path, engine._index.ntotal)
        return engine

    # ------------------------------------------------------------------
    # Stats & utilities
    # ------------------------------------------------------------------

    @property
    def stats(self) -> IndexStats:
        return IndexStats(
            num_vectors=self._index.ntotal,
            embedding_dim=self.dim,
            index_type=type(self._index).__name__,
            is_trained=self._index.is_trained,
            nlist=self.nlist,
            nprobe=self.nprobe,
        )

    def __len__(self) -> int:
        return self._index.ntotal

    def __contains__(self, track_id: str) -> bool:
        return track_id in self._rev_map

    @staticmethod
    def _validate(arr: np.ndarray) -> np.ndarray:
        """Ensure float32 C-contiguous array."""
        return np.ascontiguousarray(arr, dtype=np.float32)
