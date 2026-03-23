"""
Shared pytest fixtures and configuration.
"""

from __future__ import annotations

import io
import math
import struct
import wave

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Audio generation helpers
# ---------------------------------------------------------------------------

def generate_sine_wave(
    duration: float = 3.0,
    sr: int = 22050,
    freq: float = 440.0,
    amplitude: float = 0.5,
    noise: float = 0.0,
) -> np.ndarray:
    """Return a float32 sine wave array."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if noise > 0:
        y += (noise * np.random.randn(len(y))).astype(np.float32)
    return y


def generate_wav_bytes(
    duration: float = 3.0,
    sr: int = 22050,
    freq: float = 440.0,
) -> bytes:
    """Generate a WAV file in memory as bytes."""
    n = int(duration * sr)
    samples = [int(32767 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)]
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *samples))
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Session-level fixtures (created once per test session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_wav_bytes():
    """3-second 440Hz WAV file."""
    return generate_wav_bytes(duration=3.0)


@pytest.fixture(scope="session")
def sample_waveform():
    """3-second 440Hz float32 waveform."""
    return generate_sine_wave(duration=3.0), 22050


@pytest.fixture(scope="session")
def random_embeddings_256():
    """200 random unit-normalised 256-dim embeddings."""
    vecs = np.random.randn(200, 256).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs
