"""
Audio Feature Extractor
=======================
Extracts a comprehensive set of acoustic features from raw audio signals
using Librosa. All features are returned as a structured dictionary.

Supported features:
    - MFCCs (13–40 coefficients, with delta and delta-delta)
    - Mel Spectrogram
    - Chroma (STFT, CQT, CENS)
    - Spectral Contrast
    - Tonnetz
    - Zero-Crossing Rate
    - RMS Energy
    - Tempo & Beat frames
    - Spectral Centroid, Bandwidth, Rolloff, Flatness
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for the feature extractor."""

    sample_rate: int = 22050
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmin: float = 0.0
    fmax: float | None = None
    include_deltas: bool = True
    aggregate: bool = True  # Return mean/std instead of full matrices


@dataclass
class AudioFeatures:
    """Container for all extracted audio features."""

    mfcc: dict[str, Any] = field(default_factory=dict)
    mfcc_delta: dict[str, Any] = field(default_factory=dict)
    mfcc_delta2: dict[str, Any] = field(default_factory=dict)
    mel_spectrogram: dict[str, Any] = field(default_factory=dict)
    chroma_stft: dict[str, Any] = field(default_factory=dict)
    chroma_cqt: dict[str, Any] = field(default_factory=dict)
    chroma_cens: dict[str, Any] = field(default_factory=dict)
    spectral_contrast: dict[str, Any] = field(default_factory=dict)
    tonnetz: dict[str, Any] = field(default_factory=dict)
    zcr: dict[str, Any] = field(default_factory=dict)
    rms_energy: dict[str, Any] = field(default_factory=dict)
    spectral_centroid: dict[str, Any] = field(default_factory=dict)
    spectral_bandwidth: dict[str, Any] = field(default_factory=dict)
    spectral_rolloff: dict[str, Any] = field(default_factory=dict)
    spectral_flatness: dict[str, Any] = field(default_factory=dict)
    tempo: dict[str, Any] = field(default_factory=dict)
    sample_rate: int = 22050
    duration_seconds: float = 0.0
    extraction_time_ms: float = 0.0

    def to_flat_vector(self) -> np.ndarray:
        """
        Flatten all mean/std statistics into a single 1-D feature vector
        for use with classical ML models (SVM, RF, etc.).
        """
        parts = []
        for feat_name in [
            "mfcc", "mfcc_delta", "mfcc_delta2",
            "chroma_stft", "spectral_contrast", "tonnetz",
        ]:
            feat = getattr(self, feat_name)
            if feat:
                parts.extend(feat.get("mean", []))
                parts.extend(feat.get("std", []))
        for feat_name in ["zcr", "rms_energy", "spectral_centroid",
                          "spectral_bandwidth", "spectral_rolloff",
                          "spectral_flatness"]:
            feat = getattr(self, feat_name)
            if feat:
                parts.append(feat.get("mean", 0.0))
                parts.append(feat.get("std", 0.0))
        if self.tempo:
            parts.append(self.tempo.get("bpm", 0.0))
        return np.array(parts, dtype=np.float32)


class FeatureExtractor:
    """
    Production-grade audio feature extractor.

    Example
    -------
    >>> extractor = FeatureExtractor()
    >>> features = extractor.extract("path/to/song.mp3")
    >>> flat_vec = features.to_flat_vector()  # → shape (N,)
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.cfg = config or FeatureConfig()
        logger.info(
            "FeatureExtractor initialized | sr=%d n_mfcc=%d n_mels=%d",
            self.cfg.sample_rate, self.cfg.n_mfcc, self.cfg.n_mels,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        audio_path: str | None = None,
        y: np.ndarray | None = None,
        sr: int | None = None,
    ) -> AudioFeatures:
        """
        Extract all features from an audio file or pre-loaded waveform.

        Parameters
        ----------
        audio_path : str, optional
            Path to audio file. Used if ``y`` is not provided.
        y : np.ndarray, optional
            Pre-loaded waveform array.
        sr : int, optional
            Sample rate of ``y``. Required when ``y`` is provided.

        Returns
        -------
        AudioFeatures
        """
        t0 = time.perf_counter()

        if y is None:
            if audio_path is None:
                raise ValueError("Provide either audio_path or waveform y.")
            y, sr = librosa.load(audio_path, sr=self.cfg.sample_rate, mono=True)
        else:
            if sr is None:
                raise ValueError("sr must be provided when y is given.")
            if sr != self.cfg.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.cfg.sample_rate)
                sr = self.cfg.sample_rate

        duration = librosa.get_duration(y=y, sr=sr)
        logger.debug("Loaded audio: %.2fs at %dHz", duration, sr)

        # Shared computation: STFT magnitude
        D = np.abs(librosa.stft(y, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length))

        features = AudioFeatures(sample_rate=sr, duration_seconds=duration)

        features.mfcc = self._extract_mfcc(y, sr)
        if self.cfg.include_deltas:
            features.mfcc_delta = self._extract_mfcc_delta(y, sr, order=1)
            features.mfcc_delta2 = self._extract_mfcc_delta(y, sr, order=2)

        features.mel_spectrogram = self._extract_mel_spectrogram(y, sr)
        features.chroma_stft = self._extract_chroma_stft(D, sr)
        features.chroma_cqt = self._extract_chroma_cqt(y, sr)
        features.chroma_cens = self._extract_chroma_cens(y, sr)
        features.spectral_contrast = self._extract_spectral_contrast(D, sr)
        features.tonnetz = self._extract_tonnetz(y, sr)
        features.zcr = self._extract_zcr(y)
        features.rms_energy = self._extract_rms(y)
        features.spectral_centroid = self._extract_spectral_centroid(D, sr)
        features.spectral_bandwidth = self._extract_spectral_bandwidth(D, sr)
        features.spectral_rolloff = self._extract_spectral_rolloff(y, sr)
        features.spectral_flatness = self._extract_spectral_flatness(y)
        features.tempo = self._extract_tempo(y, sr)

        features.extraction_time_ms = (time.perf_counter() - t0) * 1000
        logger.info("Feature extraction complete in %.1fms", features.extraction_time_ms)
        return features

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _agg(self, matrix: np.ndarray) -> dict[str, Any]:
        """Aggregate a 2-D feature matrix to mean/std statistics."""
        if self.cfg.aggregate:
            return {
                "mean": matrix.mean(axis=1).tolist(),
                "std": matrix.std(axis=1).tolist(),
                "shape": list(matrix.shape),
            }
        return {"data": matrix.tolist(), "shape": list(matrix.shape)}

    def _agg_1d(self, vec: np.ndarray) -> dict[str, Any]:
        """Aggregate a 1-D feature vector."""
        return {
            "mean": float(vec.mean()),
            "std": float(vec.std()),
            "min": float(vec.min()),
            "max": float(vec.max()),
        }

    def _extract_mfcc(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=self.cfg.n_mfcc,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
        )
        return self._agg(mfcc)

    def _extract_mfcc_delta(self, y: np.ndarray, sr: int, order: int) -> dict[str, Any]:
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.cfg.n_mfcc,
            hop_length=self.cfg.hop_length,
        )
        delta = librosa.feature.delta(mfcc, order=order)
        return self._agg(delta)

    def _extract_mel_spectrogram(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=self.cfg.n_mels,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return {
            "mean": mel_db.mean(axis=1).tolist(),
            "std": mel_db.std(axis=1).tolist(),
            "shape": list(mel_db.shape),
        }

    def _extract_chroma_stft(self, D: np.ndarray, sr: int) -> dict[str, Any]:
        chroma = librosa.feature.chroma_stft(S=D**2, sr=sr, n_fft=self.cfg.n_fft)
        return self._agg(chroma)

    def _extract_chroma_cqt(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        return self._agg(chroma)

    def _extract_chroma_cens(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        return self._agg(chroma)

    def _extract_spectral_contrast(self, D: np.ndarray, sr: int) -> dict[str, Any]:
        contrast = librosa.feature.spectral_contrast(S=D, sr=sr)
        return self._agg(contrast)

    def _extract_tonnetz(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        return self._agg(tonnetz)

    def _extract_zcr(self, y: np.ndarray) -> dict[str, Any]:
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.cfg.hop_length)
        return self._agg_1d(zcr.flatten())

    def _extract_rms(self, y: np.ndarray) -> dict[str, Any]:
        rms = librosa.feature.rms(y=y, hop_length=self.cfg.hop_length)
        return self._agg_1d(rms.flatten())

    def _extract_spectral_centroid(self, D: np.ndarray, sr: int) -> dict[str, Any]:
        centroid = librosa.feature.spectral_centroid(S=D, sr=sr)
        return self._agg_1d(centroid.flatten())

    def _extract_spectral_bandwidth(self, D: np.ndarray, sr: int) -> dict[str, Any]:
        bw = librosa.feature.spectral_bandwidth(S=D, sr=sr)
        return self._agg_1d(bw.flatten())

    def _extract_spectral_rolloff(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.cfg.hop_length)
        return self._agg_1d(rolloff.flatten())

    def _extract_spectral_flatness(self, y: np.ndarray) -> dict[str, Any]:
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.cfg.hop_length)
        return self._agg_1d(flatness.flatten())

    def _extract_tempo(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.cfg.hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.cfg.hop_length)
        return {
            "bpm": float(tempo),
            "beat_frames": beat_frames.tolist(),
            "beat_times_seconds": beat_times.tolist(),
            "num_beats": int(len(beat_frames)),
        }
