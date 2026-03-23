"""
Health & Readiness Endpoints
=============================
GET /api/v1/health   — liveness probe (always 200 if server is up)
GET /api/v1/ready    — readiness probe (200 once models are loaded)
GET /api/v1/metrics  — Prometheus-compatible plaintext metrics
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import PlainTextResponse

from api.dependencies import get_mir_service
from api.schemas.audio import HealthResponse
from api.services.mir_service import MIRService

router = APIRouter()

_SERVER_START = time.monotonic()
VERSION = "1.0.0"


@router.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 as long as the API process is running.",
)
async def health(service: MIRService = Depends(get_mir_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=VERSION,
        index_size=service.index_size,
        uptime_seconds=round(service.uptime_seconds, 1),
        model_loaded=service.is_ready,
    )


@router.get(
    "/api/v1/ready",
    summary="Readiness probe",
    description="Returns 200 only after models and index are fully loaded.",
)
async def ready(service: MIRService = Depends(get_mir_service)) -> Response:
    if service.is_ready:
        return Response(content="ready", status_code=status.HTTP_200_OK)
    return Response(content="not ready", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@router.get(
    "/api/v1/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus metrics",
)
async def metrics(service: MIRService = Depends(get_mir_service)) -> str:
    """
    Expose basic counters in Prometheus exposition format.
    For full Prometheus integration, mount prometheus_client's
    make_asgi_app() at /metrics.
    """
    uptime = service.uptime_seconds
    index_size = service.index_size
    lines = [
        "# HELP mir_uptime_seconds API server uptime in seconds",
        "# TYPE mir_uptime_seconds gauge",
        f"mir_uptime_seconds {uptime:.1f}",
        "",
        "# HELP mir_index_size Number of vectors in the FAISS index",
        "# TYPE mir_index_size gauge",
        f"mir_index_size {index_size}",
        "",
        "# HELP mir_model_loaded Whether the ML model is loaded (0/1)",
        "# TYPE mir_model_loaded gauge",
        f"mir_model_loaded {1 if service.is_ready else 0}",
    ]
    return "\n".join(lines)
