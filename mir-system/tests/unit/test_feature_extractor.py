"""
Unit tests for src/mir/features/extractor.py
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mir.features.extractor import AudioFeatures, FeatureConfig, FeatureExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def silence_signal():
    """30 seconds of silence at 22050 Hz."""
    sr = 22050
    return np.zeros(sr * 30, dtype=np.float32), sr


@pytest.fixture
def sine_signal():
    """30 seconds of 440 Hz sine wave at 22050 Hz."""
    sr = 22050
    t = np.linspace(0, 30, sr * 30, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return y, sr


@pytest.fixture
def extractor():
    return FeatureExtractor(FeatureConfig(n_mfcc=13, n_mels=64, aggregate=True))


# ---------------------------------------------------------------------------
# Basic extraction tests
# ---------------------------------------------------------------------------

class TestFeatureExtractor:

    def test_extract_returns_audio_features(self, extractor, sine_signal):
        y, sr = sine_signal
        features = extractor.extract(y=y, sr=sr)
        assert isinstance(features, AudioFeatures)

    def test_mfcc_shape(self, extractor, sine_signal):
        y, sr = sine_signal
        features = extractor.extract(y=y, sr=sr)
        assert "mean" in features.mfcc
        assert "std" in features.mfcc
        assert len(features.mfcc["mean"]) == 13  # n_mfcc

    def test_mfcc_delta_included(self, extractor, sine_signal):
        y, sr = sine_signal
        features = extractor.extract(y=y, sr=sr)
        assert features.mfcc_delta
        assert features.mfcc_delta2

    def test_chroma_has_12_bins(self, extractor, sine_signal):
        y, sr = sine_signal
        features = extractor.extract(y=y, sr=sr)
        assert len(features.chroma_stft["mean"]) == 12

    def test_tempo_bpm_positive(self, extractor, sine_signal):
        y, sr = sine_signal
        features = extractor.extract(y=y, sr=sr)
        assert features.tempo["bpm"] > 0

    def test_duration_correct(self, extractor, sine_signal):
        y, sr = sine_signal
        features = extractor.extract(y=y, sr=sr)
        assert abs(features.duration_seconds - 30.0) < 0.5

    def test_extraction_time_recorded(self, extractor, sine_signal):
        y, sr = sine_signal
        features = extractor.extract(y=y, sr=sr)
        assert features.extraction_time_ms > 0

    def test_silence_does_not_crash(self, extractor, silence_signal):
        y, sr = silence_signal
        features = extractor.extract(y=y, sr=sr)
        assert features.rms_energy["mean"] < 1e-5

    def test_missing_input_raises(self, extractor):
        with pytest.raises(ValueError, match="audio_path"):
            extractor.extract()

    def test_missing_sr_raises(self, extractor, sine_signal):
        y, _ = sine_signal
        with pytest.raises(ValueError, match="sr must be provided"):
            extractor.extract(y=y)


# ---------------------------------------------------------------------------
# Flat vector tests
# ---------------------------------------------------------------------------

class TestFlatVector:

    def test_flat_vector_is_1d(self, extractor, sine_signal):
        y, sr = sine_signal
        features = extractor.extract(y=y, sr=sr)
        vec = features.to_flat_vector()
        assert vec.ndim == 1

    def test_flat_vector_dtype_float32(self, extractor, sine_signal):
        y, sr = sine_signal
        features = extractor.extract(y=y, sr=sr)
        vec = features.to_flat_vector()
        assert vec.dtype == np.float32

    def test_flat_vector_no_nan(self, extractor, sine_signal):
        y, sr = sine_signal
        features = extractor.extract(y=y, sr=sr)
        vec = features.to_flat_vector()
        assert not np.any(np.isnan(vec))

    def test_flat_vector_reproducible(self, extractor, sine_signal):
        y, sr = sine_signal
        v1 = extractor.extract(y=y, sr=sr).to_flat_vector()
        v2 = extractor.extract(y=y, sr=sr).to_flat_vector()
        np.testing.assert_array_almost_equal(v1, v2)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestFeatureConfig:

    def test_custom_n_mfcc(self, sine_signal):
        y, sr = sine_signal
        ext = FeatureExtractor(FeatureConfig(n_mfcc=20))
        features = ext.extract(y=y, sr=sr)
        assert len(features.mfcc["mean"]) == 20

    def test_no_deltas(self, sine_signal):
        y, sr = sine_signal
        ext = FeatureExtractor(FeatureConfig(include_deltas=False))
        features = ext.extract(y=y, sr=sr)
        assert not features.mfcc_delta
        assert not features.mfcc_delta2

    def test_raw_mode(self, sine_signal):
        y, sr = sine_signal
        ext = FeatureExtractor(FeatureConfig(aggregate=False, n_mfcc=13))
        features = ext.extract(y=y, sr=sr)
        assert "data" in features.mfcc
