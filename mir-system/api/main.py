"""
FastAPI Application Factory
============================
Wires together middleware, routers, startup/shutdown lifecycle,
exception handlers, and OpenAPI configuration.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.routes import audio, health
from api.middleware.rate_limit import RateLimitMiddleware

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("🎵 MIR System starting up ...")
    # Pre-load models and indexes on startup to avoid cold-start latency
    from api.dependencies import get_mir_service
    service = get_mir_service()
    await service.initialize()
    logger.info("✅ MIR System ready")
    yield
    logger.info("🛑 MIR System shutting down ...")
    await service.shutdown()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="MIR System — Music Information Retrieval API",
        description=(
            "Semantic audio analysis, feature extraction, deep-learning-based "
            "embeddings, and content-based music similarity search."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        contact={
            "name": "MIR System",
            "url": "https://github.com/yourusername/mir-system",
        },
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    )

    # ---- Middleware (order matters: outermost applied first) ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

    # ---- Request timing header ----
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

    # ---- Exception handlers ----
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc), "type": "validation_error"},
        )

    @app.exception_handler(FileNotFoundError)
    async def not_found_handler(request: Request, exc: FileNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc), "type": "not_found"},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error.", "type": "server_error"},
        )

    # ---- Routers ----
    app.include_router(health.router, tags=["Health"])
    app.include_router(audio.router, prefix="/api/v1/audio", tags=["Audio"])

    return app


app = create_app()
