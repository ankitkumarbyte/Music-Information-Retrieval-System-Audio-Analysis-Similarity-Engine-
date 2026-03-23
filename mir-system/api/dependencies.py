"""
FastAPI Dependency Injection
=============================
Provides singleton instances of shared services via FastAPI's
``Depends()`` mechanism.
"""

from __future__ import annotations

import os

from api.services.mir_service import MIRService

# Module-level singleton — created once, reused across all requests
_mir_service: MIRService | None = None


def get_mir_service() -> MIRService:
    """Return the application-wide MIR service singleton."""
    global _mir_service
    if _mir_service is None:
        _mir_service = MIRService(
            model_checkpoint=os.getenv("MODEL_CHECKPOINT", "checkpoints/encoder_best.pth"),
            faiss_index_path=os.getenv("FAISS_INDEX_PATH", "data/faiss/index.faiss"),
            device=os.getenv("MODEL_DEVICE", "cpu"),
        )
    return _mir_service
