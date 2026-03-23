"""
Model Trainer
=============
Full training pipeline for the MIR CNN encoder with:
  - Contrastive (NT-Xent) + classification losses
  - Mixed-precision training (torch.cuda.amp)
  - Learning-rate scheduling (OneCycleLR)
  - Early stopping & best-model checkpointing
  - TensorBoard / W&B logging hooks
  - Reproducible seeding
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

from src.mir.models.encoder import MIRModel, NTXentLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Paths
    output_dir: str = "checkpoints"
    log_dir: str = "logs/tensorboard"

    # Training
    epochs: int = 100
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Losses
    contrastive_weight: float = 0.7
    genre_ce_weight: float = 0.2
    instrument_bce_weight: float = 0.1
    temperature: float = 0.07

    # Scheduler
    pct_start: float = 0.3   # warm-up fraction for OneCycleLR

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Mixed precision
    use_amp: bool = True

    # Logging
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 1


# ---------------------------------------------------------------------------
# Data augmentation helpers
# ---------------------------------------------------------------------------

def augment_mel(mel: torch.Tensor) -> torch.Tensor:
    """
    Apply SpecAugment-style and amplitude augmentations to a mel spectrogram.

    Parameters
    ----------
    mel : torch.Tensor  shape (1, n_mels, T)

    Returns
    -------
    torch.Tensor  same shape, augmented
    """
    aug = mel.clone()

    # 1. Frequency masking (mask up to 20 mel bins)
    f_mask = random.randint(0, 20)
    f0 = random.randint(0, max(0, aug.shape[1] - f_mask))
    aug[:, f0:f0 + f_mask, :] = 0.0

    # 2. Time masking (mask up to 20 frames)
    t_mask = random.randint(0, 20)
    t0 = random.randint(0, max(0, aug.shape[2] - t_mask))
    aug[:, :, t0:t0 + t_mask] = 0.0

    # 3. Random amplitude scaling
    scale = random.uniform(0.8, 1.2)
    aug = aug * scale

    # 4. Gaussian noise
    aug = aug + torch.randn_like(aug) * 0.01

    return aug.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset interface (to be subclassed)
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """
    Abstract base class for MIR training datasets.

    Subclass and implement ``__len__`` and ``_load_mel``.
    ``__getitem__`` returns two augmented views (for contrastive learning)
    plus optional genre and instrument labels.
    """

    def __init__(self, mel_paths: list[str], genre_labels: list[int] | None = None,
                 instrument_labels: list[list[int]] | None = None) -> None:
        self.mel_paths = mel_paths
        self.genre_labels = genre_labels        # None → genre head not trained
        self.instrument_labels = instrument_labels  # None → instrument head not trained

    def __len__(self) -> int:
        return len(self.mel_paths)

    def __getitem__(self, idx: int) -> dict:
        mel = self._load_mel(self.mel_paths[idx])
        view1 = augment_mel(mel)
        view2 = augment_mel(mel)
        item: dict = {"view1": view1, "view2": view2}
        if self.genre_labels is not None:
            item["genre"] = torch.tensor(self.genre_labels[idx], dtype=torch.long)
        if self.instrument_labels is not None:
            item["instruments"] = torch.tensor(self.instrument_labels[idx], dtype=torch.float32)
        return item

    @staticmethod
    def _load_mel(path: str) -> torch.Tensor:
        """Load a pre-computed mel spectrogram saved as a .pt file."""
        return torch.load(path, weights_only=True)  # shape (1, n_mels, T)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float = float("inf") if mode == "min" else float("-inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        improved = (
            (value < self.best - self.min_delta) if self.mode == "min"
            else (value > self.best + self.min_delta)
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MIRTrainer:
    """
    Manages the full training lifecycle.

    Example
    -------
    >>> trainer = MIRTrainer(config)
    >>> trainer.fit(train_loader, val_loader)
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.cfg = config or TrainingConfig()
        self._seed_everything(self.cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Trainer device: %s", self.device)

        self.model = MIRModel().to(self.device)
        self.contrastive_loss = NTXentLoss(temperature=self.cfg.temperature)
        self.genre_loss = nn.CrossEntropyLoss()
        self.instrument_loss = nn.BCEWithLogitsLoss()

        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[OneCycleLR] = None
        self.scaler = GradScaler(enabled=self.cfg.use_amp)

        self.early_stopping = EarlyStopping(
            patience=self.cfg.patience,
            min_delta=self.cfg.min_delta,
        )
        self._global_step = 0
        self._best_val_loss = float("inf")

        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict:
        """Run the full training loop."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.lr,
            total_steps=len(train_loader) * self.cfg.epochs,
            pct_start=self.cfg.pct_start,
        )

        history: dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}
        t_start = time.perf_counter()

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if val_loader is not None and epoch % self.cfg.eval_every_n_epochs == 0:
                val_loss = self._eval_epoch(val_loader, epoch)
                history["val_loss"].append(val_loss)
                improved = self.early_stopping.step(val_loss)
                if improved:
                    self._save_best_checkpoint(epoch, val_loss)
                if self.early_stopping.should_stop:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break
            else:
                self._save_checkpoint(epoch, train_loss, tag="latest")

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | %.0fs elapsed",
                epoch, self.cfg.epochs, train_loss,
                time.perf_counter() - t_start,
            )

        logger.info("Training complete. Best val_loss: %.4f", self.early_stopping.best)
        return history

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader, 1):
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.cfg.use_amp):
                loss = self._compute_loss(batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            self._global_step += 1

            if step % self.cfg.log_every_n_steps == 0:
                avg = total_loss / step
                logger.debug("  Epoch %d Step %d | loss=%.4f", epoch, step, avg)

        return total_loss / len(loader)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            with autocast(enabled=self.cfg.use_amp):
                total_loss += self._compute_loss(batch).item()
        avg = total_loss / len(loader)
        logger.debug("  Val loss at epoch %d: %.4f", epoch, avg)
        return avg

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        v1 = batch["view1"].to(self.device)
        v2 = batch["view2"].to(self.device)

        out1 = self.model(v1)
        out2 = self.model(v2)

        loss = self.cfg.contrastive_weight * self.contrastive_loss(
            out1["embedding"], out2["embedding"]
        )

        if "genre" in batch:
            g = batch["genre"].to(self.device)
            loss += self.cfg.genre_ce_weight * (
                self.genre_loss(out1["genre_logits"], g) +
                self.genre_loss(out2["genre_logits"], g)
            ) / 2

        if "instruments" in batch:
            instr = batch["instruments"].to(self.device)
            loss += self.cfg.instrument_bce_weight * (
                self.instrument_loss(out1["instr_logits"], instr) +
                self.instrument_loss(out2["instr_logits"], instr)
            ) / 2

        return loss

    def _save_checkpoint(self, epoch: int, loss: float, tag: str = "latest") -> Path:
        path = Path(self.cfg.output_dir) / f"encoder_{tag}.pth"
        self.model.save(path, epoch=epoch, metadata={"loss": loss})
        return path

    def _save_best_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = self._save_checkpoint(epoch, val_loss, tag="best")
        logger.info("✅ New best model at epoch %d (val_loss=%.4f) → %s", epoch, val_loss, path)
        self._best_val_loss = val_loss

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
