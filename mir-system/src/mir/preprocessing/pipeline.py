"""
Audio Preprocessing Pipeline
=============================
End-to-end pipeline that transforms raw audio bytes / files into
clean, normalised, fixed-length waveforms ready for feature extraction.

Steps
-----
1. Format validation & multi-format loading (MP3, WAV, FLAC, OGG, M4A)
2. Mono conversion
3. Resampling to target sample rate
4. Noise reduction (spectral subtraction via noisereduce)
5. Loudness normalisation (EBU R 128)
6. Silence trimming
7. Padding / truncation to fixed duration
"""

from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = frozenset({"mp3", "wav", "flac", "ogg", "m4a", "aac", "opus", "wma"})


@dataclass
class PipelineConfig:
    """Configuration for the preprocessing pipeline."""

    sample_rate: int = 22050
    duration_seconds: float = 30.0          # Clips longer than this are truncated
    pad_short_clips: bool = True            # Pad clips shorter than duration_seconds
    mono: bool = True
    denoise: bool = True
    normalize_loudness: bool = True
    target_loudness_lufs: float = -23.0     # EBU R 128 standard
    trim_silence: bool = True
    trim_top_db: float = 60.0              # dB below max to treat as silence
    max_file_size_mb: float = 50.0


class AudioPreprocessingPipeline:
    """
    Stateless preprocessing pipeline.

    Example
    -------
    >>> pipeline = AudioPreprocessingPipeline()
    >>> y, sr = pipeline.process_file("song.mp3")
    >>> y, sr = pipeline.process_bytes(raw_bytes, filename="song.flac")
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.cfg = config or PipelineConfig()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def process_file(self, path: str | Path) -> Tuple[np.ndarray, int]:
        """Process an audio file on disk."""
        path = Path(path)
        self._validate_format(path.suffix.lstrip(".").lower())
        y, sr = self._load_file(str(path))
        return self._run_pipeline(y, sr)

    def process_bytes(self, data: bytes, filename: str) -> Tuple[np.ndarray, int]:
        """Process raw audio bytes (e.g., from an HTTP upload)."""
        ext = Path(filename).suffix.lstrip(".").lower()
        self._validate_format(ext)
        size_mb = len(data) / (1024 ** 2)
        if size_mb > self.cfg.max_file_size_mb:
            raise ValueError(
                f"File too large: {size_mb:.1f}MB > {self.cfg.max_file_size_mb}MB limit."
            )
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            y, sr = self._load_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return self._run_pipeline(y, sr)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _run_pipeline(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Execute all preprocessing stages in order."""
        logger.debug("Pipeline start | shape=%s sr=%d", y.shape, sr)

        # 1. Mono conversion
        if y.ndim > 1:
            y = librosa.to_mono(y)

        # 2. Resample
        if sr != self.cfg.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.cfg.sample_rate)
            sr = self.cfg.sample_rate

        # 3. Noise reduction
        if self.cfg.denoise:
            y = self._denoise(y, sr)

        # 4. Loudness normalisation
        if self.cfg.normalize_loudness:
            y = self._normalize_loudness(y)

        # 5. Silence trimming
        if self.cfg.trim_silence:
            y, _ = librosa.effects.trim(y, top_db=self.cfg.trim_top_db)

        # 6. Duration management
        y = self._fix_duration(y, sr)

        logger.debug("Pipeline complete | shape=%s", y.shape)
        return y, sr

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(path: str) -> Tuple[np.ndarray, int]:
        """Load audio using librosa (ffmpeg backend for broad format support)."""
        try:
            y, sr = librosa.load(path, sr=None, mono=False)
            return y, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}") from e

    def _denoise(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Spectral noise reduction.

        Uses a simple RMS-gated spectral subtraction approach when
        noisereduce is unavailable, otherwise delegates to noisereduce.
        """
        try:
            import noisereduce as nr
            return nr.reduce_noise(y=y, sr=sr, stationary=False)
        except ImportError:
            logger.debug("noisereduce not installed; using simple spectral floor.")
            return self._spectral_floor_denoise(y)

    @staticmethod
    def _spectral_floor_denoise(y: np.ndarray, percentile: float = 10.0) -> np.ndarray:
        """Minimal spectral subtraction fallback."""
        S_full = librosa.stft(y)
        magnitude = np.abs(S_full)
        noise_floor = np.percentile(magnitude, percentile, axis=1, keepdims=True)
        magnitude_clean = np.maximum(magnitude - noise_floor, 0)
        S_clean = magnitude_clean * np.exp(1j * np.angle(S_full))
        return librosa.istft(S_clean, length=len(y))

    def _normalize_loudness(self, y: np.ndarray) -> np.ndarray:
        """
        Normalise to target LUFS.

        Falls back to peak normalisation when pyloudnorm is unavailable.
        """
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(self.cfg.sample_rate)
            loudness = meter.integrated_loudness(y)
            if np.isfinite(loudness):
                y = pyln.normalize.loudness(y, loudness, self.cfg.target_loudness_lufs)
        except ImportError:
            logger.debug("pyloudnorm not installed; using peak normalisation.")
            peak = np.abs(y).max()
            if peak > 0:
                y = y / peak * 0.95
        # Clip to prevent hard clipping
        return np.clip(y, -1.0, 1.0)

    def _fix_duration(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Truncate or pad to target duration."""
        target_len = int(self.cfg.duration_seconds * sr)
        if len(y) >= target_len:
            # Centre-crop for better coverage
            start = (len(y) - target_len) // 2
            return y[start: start + target_len]
        elif self.cfg.pad_short_clips:
            pad_len = target_len - len(y)
            return np.pad(y, (0, pad_len), mode="constant")
        return y

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_format(ext: str) -> None:
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '.{ext}'. "
                f"Supported: {sorted(SUPPORTED_FORMATS)}"
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def mel_tensor(self, y: np.ndarray, sr: int, n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512):
        """
        Compute a log-mel spectrogram tensor ready for the CNN encoder.

        Returns
        -------
        torch.Tensor  shape (1, n_mels, T)
        """
        import torch
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Normalise to [0, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)
