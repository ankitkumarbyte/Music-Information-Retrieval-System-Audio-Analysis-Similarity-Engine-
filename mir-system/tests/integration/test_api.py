"""
Integration tests for the FastAPI endpoints.
Requires no external services — uses an in-memory FAISS index.
"""

from __future__ import annotations

import io
import wave
import struct
import math

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import app
from api.dependencies import get_mir_service
from api.services.mir_service import MIRService


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration: float = 2.0, sr: int = 22050, freq: float = 440.0) -> bytes:
    """Generate a synthetic WAV file (sine wave) in memory."""
    n_samples = int(duration * sr)
    samples = [
        int(32767 * math.sin(2 * math.pi * freq * i / sr))
        for i in range(n_samples)
    ]
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    buf.seek(0)
    return buf.read()


WAV_BYTES = _make_wav_bytes()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client():
    """Async HTTP client wired to the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c


@pytest.fixture(autouse=True)
def fresh_service(monkeypatch):
    """
    Reset the MIR service singleton before each test so tests are isolated.
    Uses a fast, untrained in-memory index.
    """
    import api.dependencies as deps
    service = MIRService(model_checkpoint=None, faiss_index_path=None, device="cpu")

    async def mock_init():
        from src.mir.preprocessing.pipeline import AudioPreprocessingPipeline, PipelineConfig
        from src.mir.features.extractor import FeatureExtractor, FeatureConfig
        from src.mir.models.encoder import EmbeddingInferencer, MIRModel
        from src.mir.search.faiss_index import FaissSearchEngine

        service._pipeline = AudioPreprocessingPipeline(PipelineConfig())
        service._extractor = FeatureExtractor(FeatureConfig(n_mfcc=13, n_mels=64))
        model = MIRModel()
        service._inferencer = EmbeddingInferencer(model, device="cpu")
        engine = FaissSearchEngine(dim=256, nlist=8, M=0, nprobe=4)
        dummy = np.random.randn(200, 256).astype(np.float32)
        dummy /= np.linalg.norm(dummy, axis=1, keepdims=True)
        engine.train(dummy)
        service._search = engine
        service._ready = True

    import asyncio
    asyncio.get_event_loop().run_until_complete(mock_init())

    monkeypatch.setattr(deps, "_mir_service", service)
    return service


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_schema(self, client):
        data = (await client.get("/api/v1/health")).json()
        assert "status" in data
        assert "version" in data
        assert "index_size" in data
        assert "model_loaded" in data

    @pytest.mark.asyncio
    async def test_ready_returns_200(self, client):
        resp = await client.get("/api/v1/ready")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        resp = await client.get("/api/v1/metrics")
        assert resp.status_code == 200
        assert "mir_uptime_seconds" in resp.text


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

class TestAnalyze:

    @pytest.mark.asyncio
    async def test_analyze_returns_200(self, client):
        resp = await client.post(
            "/api/v1/audio/analyze",
            files={"file": ("test.wav", WAV_BYTES, "audio/wav")},
            data={"store": "true"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_analyze_response_schema(self, client):
        data = (await client.post(
            "/api/v1/audio/analyze",
            files={"file": ("test.wav", WAV_BYTES, "audio/wav")},
        )).json()
        assert "track_id" in data
        assert "features" in data
        assert "embedding" in data
        assert "annotation" in data
        assert "processing_time_ms" in data

    @pytest.mark.asyncio
    async def test_analyze_embedding_dim(self, client):
        data = (await client.post(
            "/api/v1/audio/analyze",
            files={"file": ("test.wav", WAV_BYTES, "audio/wav")},
        )).json()
        assert len(data["embedding"]) == 256

    @pytest.mark.asyncio
    async def test_analyze_unsupported_format(self, client):
        resp = await client.post(
            "/api/v1/audio/analyze",
            files={"file": ("test.xyz", b"garbage", "application/octet-stream")},
        )
        assert resp.status_code in (415, 422)

    @pytest.mark.asyncio
    async def test_analyze_track_id_unique(self, client):
        ids = set()
        for _ in range(3):
            data = (await client.post(
                "/api/v1/audio/analyze",
                files={"file": ("test.wav", WAV_BYTES, "audio/wav")},
            )).json()
            ids.add(data["track_id"])
        assert len(ids) == 3


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

class TestSimilarity:

    @pytest.mark.asyncio
    async def test_similar_by_upload(self, client):
        # Add a track first
        analysis = (await client.post(
            "/api/v1/audio/analyze",
            files={"file": ("t.wav", WAV_BYTES, "audio/wav")},
        )).json()
        assert analysis["stored_in_index"] is True

        # Search
        resp = await client.post(
            "/api/v1/audio/similar",
            files={"file": ("q.wav", WAV_BYTES, "audio/wav")},
            data={"top_k": "5"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "search_time_ms" in data

    @pytest.mark.asyncio
    async def test_similar_by_id(self, client):
        analysis = (await client.post(
            "/api/v1/audio/analyze",
            files={"file": ("t.wav", WAV_BYTES, "audio/wav")},
        )).json()
        track_id = analysis["track_id"]

        resp = await client.get(f"/api/v1/audio/{track_id}/similar?top_k=5")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_similar_by_nonexistent_id_returns_404(self, client):
        resp = await client.get("/api/v1/audio/trk_does_not_exist/similar")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_similar_results_have_scores(self, client):
        # Add 3 tracks
        for _ in range(3):
            await client.post(
                "/api/v1/audio/analyze",
                files={"file": ("t.wav", WAV_BYTES, "audio/wav")},
            )
        data = (await client.post(
            "/api/v1/audio/similar",
            files={"file": ("q.wav", WAV_BYTES, "audio/wav")},
            data={"top_k": "3"},
        )).json()
        for r in data["results"]:
            assert 0.0 <= r["score"] <= 1.0


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

class TestCRUD:

    @pytest.mark.asyncio
    async def test_get_features(self, client):
        analysis = (await client.post(
            "/api/v1/audio/analyze",
            files={"file": ("t.wav", WAV_BYTES, "audio/wav")},
        )).json()
        resp = await client.get(f"/api/v1/audio/{analysis['track_id']}/features")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_embedding(self, client):
        analysis = (await client.post(
            "/api/v1/audio/analyze",
            files={"file": ("t.wav", WAV_BYTES, "audio/wav")},
        )).json()
        resp = await client.get(f"/api/v1/audio/{analysis['track_id']}/embedding")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["embedding"]) == 256

    @pytest.mark.asyncio
    async def test_delete_track(self, client):
        analysis = (await client.post(
            "/api/v1/audio/analyze",
            files={"file": ("t.wav", WAV_BYTES, "audio/wav")},
        )).json()
        track_id = analysis["track_id"]

        del_resp = await client.delete(f"/api/v1/audio/{track_id}")
        assert del_resp.status_code == 204

        # Should 404 after deletion
        get_resp = await client.get(f"/api/v1/audio/{track_id}/embedding")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_404(self, client):
        resp = await client.delete("/api/v1/audio/trk_ghost")
        assert resp.status_code == 404
