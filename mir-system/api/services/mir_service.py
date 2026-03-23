"""
MIR Service
===========
Orchestration layer that wires together:
  - AudioPreprocessingPipeline
  - FeatureExtractor
  - MIRModel (encoder + classifiers)
  - FaissSearchEngine

This is the single entry point for all high-level operations called by
the API routes. It is designed to be instantiated once at startup and
shared across requests (singleton via FastAPI dependency injection).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from api.schemas.audio import (
    AnalysisResponse, AudioFeaturesOut, EmbeddingResponse,
    FeaturesResponse, SimilarityResponse, SimilarTrack,
    FeatureStats, ScalarStats, TempoInfo, Annotation, GenreResult, InstrumentResult,
)
from src.mir.features.extractor import AudioFeatures, FeatureConfig, FeatureExtractor
from src.mir.models.encoder import EmbeddingInferencer, MIRModel
from src.mir.preprocessing.pipeline import AudioPreprocessingPipeline, PipelineConfig
from src.mir.search.faiss_index import FaissSearchEngine

logger = logging.getLogger(__name__)


class MIRService:
    """
    Stateful service that owns one instance of each component.

    Thread-safety: All mutation of the FAISS index is serialised by the
    engine's internal RLock. Feature extraction and model inference are
    read-only and safe for concurrent use.
    """

    def __init__(
        self,
        model_checkpoint: str | None = None,
        faiss_index_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._model_checkpoint = model_checkpoint
        self._faiss_index_path = faiss_index_path
        self._device = device

        # Components initialised in `initialize()`
        self._pipeline: Optional[AudioPreprocessingPipeline] = None
        self._extractor: Optional[FeatureExtractor] = None
        self._inferencer: Optional[EmbeddingInferencer] = None
        self._search: Optional[FaissSearchEngine] = None

        # In-memory feature / embedding store
        # Production: replace with Redis / PostgreSQL
        self._feature_store: dict[str, dict] = {}
        self._embedding_store: dict[str, list[float]] = {}

        # Batch jobs
        self._batch_jobs: dict[str, dict] = {}

        self._start_time = time.monotonic()
        self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load models and indexes (called once at API startup)."""
        logger.info("Initialising MIR service ...")

        self._pipeline = AudioPreprocessingPipeline(PipelineConfig())
        self._extractor = FeatureExtractor(FeatureConfig())

        # Load or create model
        if self._model_checkpoint and Path(self._model_checkpoint).exists():
            model = MIRModel.from_pretrained(self._model_checkpoint, device=self._device)
        else:
            logger.warning("No checkpoint found — using untrained model (for dev/demo only).")
            model = MIRModel()

        self._inferencer = EmbeddingInferencer(model, device=self._device)

        # Load or create FAISS index
        if self._faiss_index_path and Path(self._faiss_index_path).exists():
            self._search = FaissSearchEngine.load(self._faiss_index_path)
        else:
            self._search = FaissSearchEngine(dim=256, nlist=64, M=0, nprobe=16)
            # Train on random data so the index is usable immediately
            dummy = np.random.randn(500, 256).astype(np.float32)
            dummy /= np.linalg.norm(dummy, axis=1, keepdims=True)
            self._search.train(dummy)

        self._ready = True
        logger.info("MIR service ready. Index size: %d", len(self._search))

    async def shutdown(self) -> None:
        """Persist state on graceful shutdown."""
        if self._search and self._faiss_index_path:
            logger.info("Saving FAISS index ...")
            self._search.save(self._faiss_index_path)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def analyze(
        self,
        audio_bytes: bytes,
        filename: str,
        store: bool = True,
    ) -> AnalysisResponse:
        self._assert_ready()
        t0 = time.perf_counter()

        # 1. Preprocessing (CPU-bound — run in thread pool)
        y, sr = await asyncio.get_event_loop().run_in_executor(
            None, self._pipeline.process_bytes, audio_bytes, filename
        )

        # 2. Feature extraction
        features: AudioFeatures = await asyncio.get_event_loop().run_in_executor(
            None, self._extractor.extract, None, y, sr
        )

        # 3. Deep embedding + classification
        mel_tensor = self._pipeline.mel_tensor(y, sr)
        embedding_vec = await asyncio.get_event_loop().run_in_executor(
            None, self._inferencer.embed, mel_tensor
        )
        genre_result = await asyncio.get_event_loop().run_in_executor(
            None, self._inferencer.classify_genre, mel_tensor
        )
        instruments = await asyncio.get_event_loop().run_in_executor(
            None, self._inferencer.classify_instruments, mel_tensor, 0.5
        )

        # 4. Build track ID and store
        track_id = f"trk_{uuid.uuid4().hex[:8]}"
        embedding_list = embedding_vec.tolist()

        if store:
            emb_np = embedding_vec.numpy().reshape(1, -1)
            self._search.add(emb_np, [track_id])
            self._feature_store[track_id] = self._serialise_features(features)
            self._embedding_store[track_id] = embedding_list

        processing_ms = (time.perf_counter() - t0) * 1000

        return AnalysisResponse(
            track_id=track_id,
            filename=filename,
            duration_seconds=features.duration_seconds,
            sample_rate=sr,
            features=self._features_to_schema(features),
            embedding=embedding_list,
            annotation=Annotation(
                genre=GenreResult(**genre_result),
                instruments=[InstrumentResult(**i) for i in instruments],
            ),
            stored_in_index=store,
            processing_time_ms=round(processing_ms, 1),
        )

    async def find_similar_by_audio(
        self, audio_bytes: bytes, filename: str, top_k: int = 10
    ) -> SimilarityResponse:
        self._assert_ready()
        t0 = time.perf_counter()

        y, sr = await asyncio.get_event_loop().run_in_executor(
            None, self._pipeline.process_bytes, audio_bytes, filename
        )
        mel_tensor = self._pipeline.mel_tensor(y, sr)
        emb = await asyncio.get_event_loop().run_in_executor(
            None, self._inferencer.embed, mel_tensor
        )
        raw_results = self._search.search(emb.numpy(), top_k=top_k)
        search_ms = (time.perf_counter() - t0) * 1000

        return SimilarityResponse(
            results=[SimilarTrack(
                track_id=r.track_id, score=round(r.score, 4),
                distance=round(r.distance, 6), rank=i
            ) for i, r in enumerate(raw_results)],
            total_indexed=len(self._search),
            search_time_ms=round(search_ms, 1),
        )

    async def find_similar_by_id(
        self, track_id: str, top_k: int = 10
    ) -> Optional[SimilarityResponse]:
        if track_id not in self._embedding_store:
            return None

        t0 = time.perf_counter()
        emb = np.array(self._embedding_store[track_id], dtype=np.float32)
        raw_results = self._search.search(emb, top_k=top_k + 1, exclude_ids=[track_id])
        search_ms = (time.perf_counter() - t0) * 1000

        return SimilarityResponse(
            query_track_id=track_id,
            results=[SimilarTrack(
                track_id=r.track_id, score=round(r.score, 4),
                distance=round(r.distance, 6), rank=i
            ) for i, r in enumerate(raw_results[:top_k])],
            total_indexed=len(self._search),
            search_time_ms=round(search_ms, 1),
        )

    async def get_features(self, track_id: str) -> Optional[FeaturesResponse]:
        data = self._feature_store.get(track_id)
        if not data:
            return None
        return FeaturesResponse(
            track_id=track_id,
            features=AudioFeaturesOut(**data["features"]),
            duration_seconds=data["duration_seconds"],
            sample_rate=data["sample_rate"],
        )

    async def get_embedding(self, track_id: str) -> Optional[EmbeddingResponse]:
        emb = self._embedding_store.get(track_id)
        if emb is None:
            return None
        return EmbeddingResponse(track_id=track_id, embedding=emb, embedding_dim=len(emb))

    async def delete_track(self, track_id: str) -> bool:
        removed = self._search.remove([track_id])
        self._feature_store.pop(track_id, None)
        self._embedding_store.pop(track_id, None)
        return removed > 0

    async def create_batch_job(self, batch_data: list[tuple[bytes, str]]) -> str:
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        self._batch_jobs[batch_id] = {
            "status": "queued",
            "total": len(batch_data),
            "completed": 0,
            "failed": 0,
            "results": [],
            "data": batch_data,
        }
        return batch_id

    async def process_batch(self, batch_id: str) -> None:
        job = self._batch_jobs.get(batch_id)
        if not job:
            return
        job["status"] = "processing"
        for audio_bytes, filename in job["data"]:
            try:
                result = await self.analyze(audio_bytes, filename, store=True)
                job["results"].append({"track_id": result.track_id, "filename": filename})
                job["completed"] += 1
            except Exception as e:
                logger.error("Batch %s: failed on %s: %s", batch_id, filename, e)
                job["failed"] += 1
        job["status"] = "completed"
        del job["data"]   # Free memory

    # ------------------------------------------------------------------
    # Properties for health / monitoring
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def index_size(self) -> int:
        return len(self._search) if self._search else 0

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("MIR service is not initialised. Call initialize() first.")

    @staticmethod
    def _serialise_features(f: AudioFeatures) -> dict:
        return {
            "duration_seconds": f.duration_seconds,
            "sample_rate": f.sample_rate,
            "features": {
                "mfcc": f.mfcc or None,
                "mfcc_delta": f.mfcc_delta or None,
                "mel_spectrogram": f.mel_spectrogram or None,
                "chroma_stft": f.chroma_stft or None,
                "spectral_contrast": f.spectral_contrast or None,
                "tonnetz": f.tonnetz or None,
                "zcr": f.zcr or None,
                "rms_energy": f.rms_energy or None,
                "spectral_centroid": f.spectral_centroid or None,
                "spectral_bandwidth": f.spectral_bandwidth or None,
                "spectral_rolloff": f.spectral_rolloff or None,
                "spectral_flatness": f.spectral_flatness or None,
                "tempo": f.tempo or None,
            },
        }

    @staticmethod
    def _features_to_schema(f: AudioFeatures) -> AudioFeaturesOut:
        def to_stats(d: dict | None):
            if not d or "mean" not in d:
                return None
            return FeatureStats(mean=d["mean"], std=d["std"], shape=d.get("shape", []))

        def to_scalar(d: dict | None):
            if not d:
                return None
            return ScalarStats(
                mean=d.get("mean", 0.0), std=d.get("std", 0.0),
                min=d.get("min", 0.0), max=d.get("max", 0.0),
            )

        tempo = None
        if f.tempo:
            tempo = TempoInfo(
                bpm=f.tempo.get("bpm", 0.0),
                num_beats=f.tempo.get("num_beats", 0),
                beat_times_seconds=f.tempo.get("beat_times_seconds", []),
            )

        return AudioFeaturesOut(
            mfcc=to_stats(f.mfcc),
            mfcc_delta=to_stats(f.mfcc_delta),
            mfcc_delta2=to_stats(f.mfcc_delta2),
            mel_spectrogram=to_stats(f.mel_spectrogram),
            chroma_stft=to_stats(f.chroma_stft),
            chroma_cqt=to_stats(f.chroma_cqt),
            chroma_cens=to_stats(f.chroma_cens),
            spectral_contrast=to_stats(f.spectral_contrast),
            tonnetz=to_stats(f.tonnetz),
            zcr=to_scalar(f.zcr),
            rms_energy=to_scalar(f.rms_energy),
            spectral_centroid=to_scalar(f.spectral_centroid),
            spectral_bandwidth=to_scalar(f.spectral_bandwidth),
            spectral_rolloff=to_scalar(f.spectral_rolloff),
            spectral_flatness=to_scalar(f.spectral_flatness),
            tempo=tempo,
        )
