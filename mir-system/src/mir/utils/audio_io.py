"""
Audio I/O Utilities
====================
Thin wrappers around librosa / soundfile for consistent
multi-format audio loading with error handling and logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


def load_audio(
    path: str | Path,
    sr: int | None = 22050,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file into a NumPy waveform.

    Parameters
    ----------
    path     : file path (any format supported by librosa / ffmpeg)
    sr       : target sample rate; None = keep native SR
    mono     : mix to mono if True
    offset   : start reading at this many seconds
    duration : only load this many seconds (None = full file)

    Returns
    -------
    (y, sr)  waveform array and actual sample rate
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Extension '{ext}' not supported. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        y, actual_sr = librosa.load(
            str(path), sr=sr, mono=mono, offset=offset, duration=duration
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load '{path}': {exc}") from exc

    logger.debug("Loaded '%s' | sr=%d shape=%s", path.name, actual_sr, y.shape)
    return y, actual_sr


def save_audio(
    y: np.ndarray,
    path: str | Path,
    sr: int = 22050,
    fmt: str = "WAV",
    subtype: str = "PCM_16",
) -> None:
    """
    Save a waveform to disk.

    Parameters
    ----------
    y    : float32 or float64 waveform, mono (N,) or stereo (N, 2)
    path : output path; extension determines format if fmt is not set
    sr   : sample rate
    fmt  : soundfile format string ("WAV", "FLAC", "OGG")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.T if y.ndim > 1 else y, sr, format=fmt, subtype=subtype)
    logger.debug("Saved audio to '%s'", path)


def get_duration(path: str | Path) -> float:
    """Return audio duration in seconds without loading the full file."""
    return librosa.get_duration(path=str(path))


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample waveform to target sample rate."""
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def is_supported(path: str | Path) -> bool:
    """Return True if the file extension is in the supported list."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS
