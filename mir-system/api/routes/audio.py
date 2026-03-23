"""
Audio Routes
============
POST /api/v1/audio/analyze         — Upload audio, get full analysis
POST /api/v1/audio/similar         — Find similar tracks by audio
GET  /api/v1/audio/{track_id}/similar — Similar tracks by stored ID
GET  /api/v1/audio/{track_id}/features — Retrieve stored features
GET  /api/v1/audio/{track_id}/embedding — Get stored embedding
POST /api/v1/audio/batch           — Bulk upload & analysis
DELETE /api/v1/audio/{track_id}    — Remove track from index
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status

from api.dependencies import get_mir_service
from api.schemas.audio import (
    AnalysisResponse,
    BatchUploadResponse,
    EmbeddingResponse,
    FeaturesResponse,
    SimilarityResponse,
)
from api.services.mir_service import MIRService

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_audio_file(file: UploadFile) -> None:
    allowed = {"audio/mpeg", "audio/wav", "audio/flac", "audio/ogg",
               "audio/mp4", "audio/x-m4a", "audio/aac", "audio/opus"}
    ct = file.content_type or ""
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    allowed_exts = {"mp3", "wav", "flac", "ogg", "m4a", "aac", "opus"}
    if ct not in allowed and ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: '{ct}'. Upload MP3, WAV, FLAC, OGG, or M4A.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and analyse an audio file",
    description=(
        "Upload an audio file (MP3, WAV, FLAC, OGG, M4A) and receive a complete "
        "acoustic analysis including MFCC features, mel spectrogram statistics, "
        "chroma, tempo, a 256-dim deep embedding, genre classification, and "
        "instrument recognition."
    ),
)
async def analyze_audio(
    file: Annotated[UploadFile, File(description="Audio file to analyse")],
    store: Annotated[bool, Form(description="Persist track in the similarity index")] = True,
    service: MIRService = Depends(get_mir_service),
) -> AnalysisResponse:
    _check_audio_file(file)
    logger.info("Received audio file: %s (%.1f KB)", file.filename, (file.size or 0) / 1024)
    data = await file.read()
    try:
        result = await service.analyze(
            audio_bytes=data,
            filename=file.filename or "upload.mp3",
            store=store,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return result


@router.post(
    "/similar",
    response_model=SimilarityResponse,
    status_code=status.HTTP_200_OK,
    summary="Find similar tracks by audio upload",
)
async def find_similar_by_upload(
    file: Annotated[UploadFile, File(description="Query audio file")],
    top_k: Annotated[int, Form(description="Number of results", ge=1, le=100)] = 10,
    service: MIRService = Depends(get_mir_service),
) -> SimilarityResponse:
    _check_audio_file(file)
    data = await file.read()
    try:
        result = await service.find_similar_by_audio(
            audio_bytes=data,
            filename=file.filename or "query.mp3",
            top_k=top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return result


@router.get(
    "/{track_id}/similar",
    response_model=SimilarityResponse,
    status_code=status.HTTP_200_OK,
    summary="Find similar tracks by stored track ID",
)
async def find_similar_by_id(
    track_id: str,
    top_k: int = 10,
    service: MIRService = Depends(get_mir_service),
) -> SimilarityResponse:
    if top_k < 1 or top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100.")
    result = await service.find_similar_by_id(track_id=track_id, top_k=top_k)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found in index.")
    return result


@router.get(
    "/{track_id}/features",
    response_model=FeaturesResponse,
    summary="Retrieve stored audio features",
)
async def get_features(
    track_id: str,
    service: MIRService = Depends(get_mir_service),
) -> FeaturesResponse:
    result = await service.get_features(track_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found.")
    return result


@router.get(
    "/{track_id}/embedding",
    response_model=EmbeddingResponse,
    summary="Retrieve the 256-dim embedding vector for a track",
)
async def get_embedding(
    track_id: str,
    service: MIRService = Depends(get_mir_service),
) -> EmbeddingResponse:
    result = await service.get_embedding(track_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found.")
    return result


@router.post(
    "/batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Bulk audio upload and analysis (async background task)",
)
async def batch_upload(
    files: Annotated[list[UploadFile], File(description="Audio files (max 20)")],
    background_tasks: BackgroundTasks,
    service: MIRService = Depends(get_mir_service),
) -> BatchUploadResponse:
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch request.")
    for f in files:
        _check_audio_file(f)

    batch_data = [(await f.read(), f.filename or f"file_{i}.mp3") for i, f in enumerate(files)]
    batch_id = await service.create_batch_job(batch_data)
    background_tasks.add_task(service.process_batch, batch_id)
    return BatchUploadResponse(
        batch_id=batch_id,
        num_files=len(files),
        status="queued",
        message="Batch job created. Poll /api/v1/audio/batch/{batch_id} for status.",
    )


@router.delete(
    "/{track_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a track from the similarity index",
)
async def delete_track(
    track_id: str,
    service: MIRService = Depends(get_mir_service),
) -> None:
    removed = await service.delete_track(track_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found.")
