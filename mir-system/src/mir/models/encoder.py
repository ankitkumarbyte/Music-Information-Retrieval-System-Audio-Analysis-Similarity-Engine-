"""
CNN Audio Encoder
=================
Deep learning model that encodes mel spectrograms into a dense,
L2-normalised 256-dimensional embedding space.

Architecture summary
--------------------
Input  : (B, 1, 128, 256)  — batch of mono mel spectrograms
Output : (B, 256)          — L2-normalised embeddings

Training: SimCLR-style contrastive learning with NT-Xent loss.
Genre / instrument classification heads branch off the penultimate layer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU → optional MaxPool."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    """Lightweight residual block for deeper architectures."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


# ---------------------------------------------------------------------------
# Encoder Backbone
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """
    CNN encoder that maps a mel spectrogram to a 256-dim embedding.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the output embedding (default 256).
    dropout : float
        Dropout probability in the MLP head.
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # ---- Convolutional backbone ----
        self.backbone = nn.Sequential(
            ConvBlock(1, 32, pool=True),          # → (B, 32, 64, 128)
            ResidualBlock(32),
            ConvBlock(32, 64, pool=True),          # → (B, 64, 32, 64)
            ResidualBlock(64),
            ConvBlock(64, 128, pool=True),         # → (B, 128, 16, 32)
            ResidualBlock(128),
            ConvBlock(128, 256, pool=True),        # → (B, 256, 8, 16)
            ResidualBlock(256),
            nn.AdaptiveAvgPool2d((1, 1)),          # → (B, 256, 1, 1)
        )

        # ---- Projection / embedding head ----
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 1, n_mels, T)

        Returns
        -------
        torch.Tensor  shape (B, embedding_dim)  — L2 normalised
        """
        features = self.backbone(x)
        embeddings = self.projector(features)
        return F.normalize(embeddings, p=2, dim=1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Full MIR Model (Encoder + Classification Heads)
# ---------------------------------------------------------------------------

class MIRModel(nn.Module):
    """
    Complete model with:
      - Shared CNN encoder backbone
      - Contrastive embedding head (256-dim)
      - Genre classification head (multi-class)
      - Instrument detection head (multi-label)

    Parameters
    ----------
    num_genres : int
        Number of genre classes (default 10 for GTZAN).
    num_instruments : int
        Number of instrument classes (default 16).
    embedding_dim : int
        Embedding dimensionality.
    dropout : float
        Dropout probability.
    """

    GENRES = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock",
    ]
    INSTRUMENTS = [
        "piano", "guitar", "bass", "drums", "violin",
        "cello", "trumpet", "saxophone", "flute", "synthesizer",
        "drum_machine", "vocals", "organ", "harmonica", "banjo", "mandolin",
    ]

    def __init__(
        self,
        num_genres: int = 10,
        num_instruments: int = 16,
        embedding_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = AudioEncoder(embedding_dim=embedding_dim, dropout=dropout)

        # Classification heads (branch from pre-normalisation features)
        self.genre_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_genres),
        )
        self.instrument_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_instruments),
        )

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            "embedding"   : (B, 256)  L2-normalised
            "genre_logits": (B, num_genres)
            "instr_logits": (B, num_instruments)
        """
        embedding = self.encoder(x)
        return {
            "embedding": embedding,
            "genre_logits": self.genre_head(embedding),
            "instr_logits": self.instrument_head(embedding),
        }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path, device: str = "cpu") -> "MIRModel":
        """Load a model from a .pth checkpoint file."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get("config", {})
        model = cls(
            num_genres=config.get("num_genres", 10),
            num_instruments=config.get("num_instruments", 16),
            embedding_dim=config.get("embedding_dim", 256),
            dropout=config.get("dropout", 0.3),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info("Loaded checkpoint from %s (epoch %d)", checkpoint_path, checkpoint.get("epoch", -1))
        return model

    def save(self, path: str | Path, epoch: int = 0, metadata: dict | None = None) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "config": {
                "embedding_dim": self.encoder.embedding_dim,
            },
            **(metadata or {}),
        }, path)
        logger.info("Saved checkpoint to %s", path)


# ---------------------------------------------------------------------------
# NT-Xent Contrastive Loss
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy Loss (SimCLR).

    Parameters
    ----------
    temperature : float
        Softmax temperature τ (default 0.07).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_i, z_j : torch.Tensor  shape (B, D)
            L2-normalised embeddings of two augmented views.
        """
        B = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B)

        # Mask self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

class EmbeddingInferencer:
    """
    Stateless helper for embedding inference (no grad, batched).

    Example
    -------
    >>> model = MIRModel.from_pretrained("checkpoints/encoder_v1.pth")
    >>> inf = EmbeddingInferencer(model, device="cuda")
    >>> emb = inf.embed(mel_spectrogram_tensor)  # (256,)
    """

    def __init__(self, model: MIRModel, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mel : torch.Tensor  shape (1, n_mels, T) or (B, 1, n_mels, T)

        Returns
        -------
        torch.Tensor  shape (256,) or (B, 256)
        """
        if mel.dim() == 3:
            mel = mel.unsqueeze(0)
        mel = mel.to(self.device)
        out = self.model(mel)
        return out["embedding"].squeeze(0).cpu()

    @torch.no_grad()
    def classify_genre(self, mel: torch.Tensor) -> dict[str, object]:
        """Return genre label and top-3 probabilities."""
        if mel.dim() == 3:
            mel = mel.unsqueeze(0)
        out = self.model(mel.to(self.device))
        probs = torch.softmax(out["genre_logits"], dim=1).squeeze(0).cpu()
        top3 = probs.topk(3)
        return {
            "label": MIRModel.GENRES[int(probs.argmax())],
            "confidence": float(probs.max()),
            "top_3": [
                {"label": MIRModel.GENRES[int(idx)], "score": float(score)}
                for score, idx in zip(top3.values, top3.indices)
            ],
        }

    @torch.no_grad()
    def classify_instruments(
        self, mel: torch.Tensor, threshold: float = 0.5
    ) -> list[dict[str, object]]:
        """Return instruments with probability above threshold."""
        if mel.dim() == 3:
            mel = mel.unsqueeze(0)
        out = self.model(mel.to(self.device))
        probs = torch.sigmoid(out["instr_logits"]).squeeze(0).cpu()
        detected = [
            {"label": MIRModel.INSTRUMENTS[i], "confidence": float(probs[i])}
            for i in range(len(MIRModel.INSTRUMENTS))
            if probs[i] >= threshold
        ]
        return sorted(detected, key=lambda d: d["confidence"], reverse=True)
