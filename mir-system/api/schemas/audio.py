"""
API Request / Response Schemas
================================
All Pydantic v2 models used by the FastAPI routes.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / nested
# ---------------------------------------------------------------------------

class FeatureStats(BaseModel):
    """Aggregated statistics for a 2-D feature matrix."""
    mean: list[float]
    std: list[float]
    shape: list[int]


class ScalarStats(BaseModel):
    """Aggregated statistics for a 1-D feature."""
    mean: float
    std: float
    min: float
    max: float


class TempoInfo(BaseModel):
    bpm: float
    num_beats: int
    beat_times_seconds: list[float] = Field(default_factory=list)


class GenreResult(BaseModel):
    label: str
    confidence: float
    top_3: list[dict[str, Any]] = Field(default_factory=list)


class InstrumentResult(BaseModel):
    label: str
    confidence: float


class Annotation(BaseModel):
    genre: GenreResult
    instruments: list[InstrumentResult] = Field(default_factory=list)
    mood: Optional[str] = None
    key: Optional[str] = None
    time_signature: Optional[str] = None


class AudioFeaturesOut(BaseModel):
    mfcc: Optional[FeatureStats] = None
    mfcc_delta: Optional[FeatureStats] = None
    mfcc_delta2: Optional[FeatureStats] = None
    mel_spectrogram: Optional[FeatureStats] = None
    chroma_stft: Optional[FeatureStats] = None
    chroma_cqt: Optional[FeatureStats] = None
    chroma_cens: Optional[FeatureStats] = None
    spectral_contrast: Optional[FeatureStats] = None
    tonnetz: Optional[FeatureStats] = None
    zcr: Optional[ScalarStats] = None
    rms_energy: Optional[ScalarStats] = None
    spectral_centroid: Optional[ScalarStats] = None
    spectral_bandwidth: Optional[ScalarStats] = None
    spectral_rolloff: Optional[ScalarStats] = None
    spectral_flatness: Optional[ScalarStats] = None
    tempo: Optional[TempoInfo] = None


class SimilarTrack(BaseModel):
    track_id: str
    score: float = Field(ge=0.0, le=1.0, description="Cosine similarity [0, 1]")
    distance: float
    rank: int


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    track_id: str
    filename: str
    duration_seconds: float
    sample_rate: int
    features: AudioFeaturesOut
    embedding: list[float] = Field(description="256-dimensional L2-normalised embedding")
    annotation: Annotation
    stored_in_index: bool
    processing_time_ms: float

    model_config = {"json_schema_extra": {
        "example": {
            "track_id": "trk_9f3a2b1c",
            "filename": "mysong.mp3",
            "duration_seconds": 213.4,
            "sample_rate": 22050,
            "stored_in_index": True,
            "processing_time_ms": 847.3,
        }
    }}


class SimilarityResponse(BaseModel):
    query_track_id: Optional[str] = None
    results: list[SimilarTrack]
    total_indexed: int
    search_time_ms: float


class FeaturesResponse(BaseModel):
    track_id: str
    features: AudioFeaturesOut
    duration_seconds: float
    sample_rate: int


class EmbeddingResponse(BaseModel):
    track_id: str
    embedding: list[float] = Field(description="256-dimensional embedding")
    embedding_dim: int = 256


class BatchUploadResponse(BaseModel):
    batch_id: str
    num_files: int
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    index_size: int
    uptime_seconds: float
    model_loaded: bool
