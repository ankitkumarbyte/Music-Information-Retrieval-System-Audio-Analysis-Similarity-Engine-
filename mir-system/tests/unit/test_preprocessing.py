"""
Unit tests for src/mir/preprocessing/pipeline.py
"""

from __future__ import annotations

import io
import math
import struct
import wave

import numpy as np
import pytest

from src.mir.preprocessing.pipeline import AudioPreprocessingPipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sine_wav(duration: float = 3.0, sr: int = 22050, freq: float = 440.0) -> bytes:
    """Generate an in-memory WAV file with a pure sine wave."""
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


@pytest.fixture
def pipeline():
    return AudioPreprocessingPipeline(PipelineConfig(
        denoise=False,           # skip heavy ops in unit tests
        normalize_loudness=True,
        trim_silence=False,
        duration_seconds=3.0,
        pad_short_clips=True,
    ))


@pytest.fixture
def wav_bytes():
    return _sine_wav(duration=3.0)


@pytest.fixture
def short_wav_bytes():
    return _sine_wav(duration=1.0)


@pytest.fixture
def long_wav_bytes():
    return _sine_wav(duration=10.0)


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

class TestProcessBytes:

    def test_returns_tuple(self, pipeline, wav_bytes):
        result = pipeline.process_bytes(wav_bytes, "test.wav")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_sample_rate(self, pipeline, wav_bytes):
        y, sr = pipeline.process_bytes(wav_bytes, "test.wav")
        assert sr == 22050

    def test_output_is_float32(self, pipeline, wav_bytes):
        y, sr = pipeline.process_bytes(wav_bytes, "test.wav")
        assert y.dtype == np.float32

    def test_output_is_mono(self, pipeline, wav_bytes):
        y, sr = pipeline.process_bytes(wav_bytes, "test.wav")
        assert y.ndim == 1

    def test_amplitude_clipped(self, pipeline, wav_bytes):
        y, _ = pipeline.process_bytes(wav_bytes, "test.wav")
        assert y.max() <= 1.0
        assert y.min() >= -1.0

    def test_no_nan_or_inf(self, pipeline, wav_bytes):
        y, _ = pipeline.process_bytes(wav_bytes, "test.wav")
        assert not np.isnan(y).any()
        assert not np.isinf(y).any()


class TestDurationHandling:

    def test_long_clip_truncated(self, pipeline, long_wav_bytes):
        y, sr = pipeline.process_bytes(long_wav_bytes, "long.wav")
        expected_len = int(3.0 * sr)
        assert abs(len(y) - expected_len) <= sr // 10  # within 100ms

    def test_short_clip_padded(self, pipeline, short_wav_bytes):
        y, sr = pipeline.process_bytes(short_wav_bytes, "short.wav")
        expected_len = int(3.0 * sr)
        assert len(y) == expected_len

    def test_no_padding_when_disabled(self, short_wav_bytes):
        pipe = AudioPreprocessingPipeline(PipelineConfig(
            pad_short_clips=False, denoise=False, duration_seconds=3.0
        ))
        y, sr = pipe.process_bytes(short_wav_bytes, "short.wav")
        # Should be shorter than target duration
        assert len(y) < int(3.0 * sr)


class TestValidation:

    def test_unsupported_format_raises(self, pipeline):
        with pytest.raises(ValueError, match="Unsupported format"):
            pipeline.process_bytes(b"data", "audio.xyz")

    def test_file_too_large_raises(self):
        pipe = AudioPreprocessingPipeline(PipelineConfig(max_file_size_mb=0.0001))
        wav = _sine_wav(1.0)
        with pytest.raises(ValueError, match="too large"):
            pipe.process_bytes(wav, "big.wav")


class TestMelTensor:

    def test_mel_tensor_shape(self, pipeline, wav_bytes):
        import torch
        y, sr = pipeline.process_bytes(wav_bytes, "test.wav")
        tensor = pipeline.mel_tensor(y, sr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 1   # channel dim
        assert tensor.shape[1] == 128  # n_mels default

    def test_mel_tensor_range(self, pipeline, wav_bytes):
        y, sr = pipeline.process_bytes(wav_bytes, "test.wav")
        tensor = pipeline.mel_tensor(y, sr)
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) <= 1.0 + 1e-5
