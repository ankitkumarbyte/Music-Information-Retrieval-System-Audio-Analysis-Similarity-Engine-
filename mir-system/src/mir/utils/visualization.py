"""
Visualization Utilities
========================
Functions to visualize audio features, spectrograms, and embedding spaces.
Used for analysis, debugging, and documentation.

All plot functions return a ``matplotlib.figure.Figure`` object so callers
can either display or save them.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")   # non-interactive backend (safe for servers)

COLORMAP = "magma"


# ---------------------------------------------------------------------------
# Spectrogram plots
# ---------------------------------------------------------------------------

def plot_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
    title: str = "Mel Spectrogram",
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Plot a mel spectrogram in dB scale."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=ax, cmap=COLORMAP,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (mel)")
    fig.tight_layout()
    return fig


def plot_waveform(
    y: np.ndarray,
    sr: int,
    title: str = "Waveform",
    figsize: tuple = (12, 3),
) -> plt.Figure:
    """Plot raw waveform amplitude over time."""
    times = np.linspace(0, len(y) / sr, num=len(y))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, y, color="#4C72B0", linewidth=0.5, alpha=0.8)
    ax.fill_between(times, y, alpha=0.3, color="#4C72B0")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, times[-1])
    fig.tight_layout()
    return fig


def plot_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    hop_length: int = 512,
    title: str = "MFCCs",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot MFCC heatmap."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        mfcc, sr=sr, hop_length=hop_length,
        x_axis="time", ax=ax, cmap="coolwarm",
    )
    fig.colorbar(img, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC Coefficient")
    fig.tight_layout()
    return fig


def plot_chroma(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    title: str = "Chroma Features",
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Plot chromagram."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        chroma, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="chroma", ax=ax, cmap="Greens",
    )
    fig.colorbar(img, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_feature_dashboard(
    y: np.ndarray,
    sr: int,
    title: str = "Audio Feature Dashboard",
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    4-panel dashboard: waveform, mel spectrogram, MFCCs, chroma.
    Great for a quick visual overview of any audio track.
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    # 1. Waveform
    times = np.linspace(0, len(y) / sr, num=len(y))
    axes[0].plot(times, y, color="#2E86AB", linewidth=0.4, alpha=0.85)
    axes[0].fill_between(times, y, alpha=0.2, color="#2E86AB")
    axes[0].set_title("Waveform", fontsize=12)
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    # 2. Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img2 = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                                     ax=axes[1], cmap=COLORMAP)
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")
    axes[1].set_title("Mel Spectrogram", fontsize=12)

    # 3. MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    img3 = librosa.display.specshow(mfcc, sr=sr, x_axis="time",
                                     ax=axes[2], cmap="coolwarm")
    fig.colorbar(img3, ax=axes[2])
    axes[2].set_title("MFCCs (20 coefficients)", fontsize=12)
    axes[2].set_ylabel("Coefficient")

    # 4. Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    img4 = librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma",
                                     ax=axes[3], cmap="Greens")
    fig.colorbar(img4, ax=axes[3])
    axes[3].set_title("Chromagram", fontsize=12)
    axes[3].set_xlabel("Time (s)")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Embedding visualization
# ---------------------------------------------------------------------------

def plot_embedding_scatter(
    embeddings: np.ndarray,
    labels: list[str],
    label_names: Optional[list[str]] = None,
    method: str = "tsne",
    title: str = "Embedding Space",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    2-D scatter of high-dim embeddings reduced via t-SNE or UMAP.

    Parameters
    ----------
    embeddings : np.ndarray  shape (N, D)
    labels     : list[str|int]  class label for each point
    label_names: list[str]  human-readable names for label indices
    method     : "tsne" | "umap"
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)
    unique_labels = le.classes_

    # Dimensionality reduction
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'tsne' or 'umap'.")

    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = label_ids == i
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)], label=label_names[i] if label_names else str(label),
            alpha=0.7, s=25, edgecolors="none",
        )

    ax.legend(
        title="Class", bbox_to_anchor=(1.05, 1), loc="upper left",
        framealpha=0.9, fontsize=9,
    )
    ax.set_title(f"{title} ({method.upper()})", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} dim 1")
    ax.set_ylabel(f"{method.upper()} dim 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_similarity_heatmap(
    embeddings: np.ndarray,
    track_ids: list[str],
    figsize: tuple = (10, 9),
    title: str = "Pairwise Cosine Similarity",
) -> plt.Figure:
    """Plot an N×N cosine similarity heatmap for a set of tracks."""
    # L2-normalise then dot product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Cosine Similarity")

    n = len(track_ids)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(track_ids, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(track_ids, fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def fig_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 150) -> bytes:
    """Serialise a Matplotlib figure to raw PNG/SVG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def save_fig(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save figure to disk and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
