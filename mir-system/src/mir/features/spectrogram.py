"""
Spectrogram Computation
========================
Functions for computing and transforming spectrograms
used as inputs to the CNN encoder.
"""

from __future__ import annotations

import librosa
import numpy as np
import torch


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: float | None = None,
    to_db: bool = True,
    normalise: bool = True,
) -> np.ndarray:
    """
    Compute a mel spectrogram.

    Parameters
    ----------
    y         : waveform
    sr        : sample rate
    to_db     : convert power to dB scale
    normalise : normalise to [0, 1] range

    Returns
    -------
    np.ndarray  shape (n_mels, T)
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=fmin, fmax=fmax,
    )
    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    if normalise:
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    return mel.astype(np.float32)


def mel_to_tensor(mel: np.ndarray) -> torch.Tensor:
    """
    Convert a mel spectrogram array to a PyTorch tensor.

    Parameters
    ----------
    mel : np.ndarray  shape (n_mels, T)

    Returns
    -------
    torch.Tensor  shape (1, n_mels, T)  — channel-first
    """
    return torch.tensor(mel, dtype=torch.float32).unsqueeze(0)


def fixed_length_mel(
    y: np.ndarray,
    sr: int,
    duration_seconds: float = 30.0,
    n_mels: int = 128,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute a mel spectrogram with a fixed number of time frames.
    Pads or truncates to match ``duration_seconds``.

    Returns
    -------
    np.ndarray  shape (n_mels, T_fixed)
    """
    target_frames = int(duration_seconds * sr / hop_length) + 1
    mel = compute_mel_spectrogram(y, sr, n_mels=n_mels, hop_length=hop_length)
    T = mel.shape[1]
    if T >= target_frames:
        mel = mel[:, :target_frames]
    else:
        mel = np.pad(mel, ((0, 0), (0, target_frames - T)), mode="constant")
    return mel


def compute_cqt(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
) -> np.ndarray:
    """Compute a Constant-Q Transform magnitude in dB."""
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length,
                            n_bins=n_bins, bins_per_octave=bins_per_octave))
    return librosa.amplitude_to_db(C, ref=np.max)
