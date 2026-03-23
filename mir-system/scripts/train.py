#!/usr/bin/env python3
"""
Train the MIR CNN Encoder
=========================
Usage
-----
    python scripts/train.py \\
        --data_dir  data/mel_spectrograms \\
        --genre_csv data/genres.csv \\
        --output    checkpoints/ \\
        --epochs    100 \\
        --batch     64 \\
        --device    cuda

Data format
-----------
data_dir/
    track_001.pt    ← pre-computed mel spectrogram tensor (1, 128, 256)
    track_002.pt
    ...

genre_csv columns: filename, genre_id  (int 0–9)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, random_split

from src.mir.models.encoder import MIRModel
from src.mir.models.trainer import AudioDataset, MIRTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MIR CNN encoder")
    p.add_argument("--data_dir", required=True, help="Directory of .pt mel spectrograms")
    p.add_argument("--genre_csv", default=None, help="CSV with (filename, genre_id) columns")
    p.add_argument("--output", default="checkpoints", help="Checkpoint output directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.1, help="Fraction for validation")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed-precision training")
    p.add_argument("--resume", default=None, help="Resume from checkpoint path")
    return p.parse_args()


def load_genre_labels(genre_csv: str, mel_paths: list[str]) -> list[int] | None:
    """Load genre labels aligned to mel_paths list."""
    if not genre_csv or not Path(genre_csv).exists():
        return None
    label_map: dict[str, int] = {}
    with open(genre_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_map[row["filename"]] = int(row["genre_id"])
    labels = []
    for path in mel_paths:
        fname = Path(path).name
        labels.append(label_map.get(fname, 0))
    return labels


def main() -> None:
    args = parse_args()

    # ---- Device ----
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Training on device: %s", device)

    # ---- Discover data ----
    data_dir = Path(args.data_dir)
    mel_paths = sorted(str(p) for p in data_dir.glob("*.pt"))
    if not mel_paths:
        logger.error("No .pt files found in %s", data_dir)
        sys.exit(1)
    logger.info("Found %d mel spectrograms", len(mel_paths))

    # ---- Labels ----
    genre_labels = load_genre_labels(args.genre_csv, mel_paths)
    if genre_labels:
        logger.info("Genre labels loaded for %d tracks", len(genre_labels))
    else:
        logger.info("No genre labels — training with contrastive loss only")

    # ---- Dataset split ----
    full_dataset = AudioDataset(mel_paths, genre_labels=genre_labels)
    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    logger.info("Split: %d train / %d val", train_size, val_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # ---- Config ----
    cfg = TrainingConfig(
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.workers,
        use_amp=not args.no_amp,
    )

    # ---- Trainer ----
    trainer = MIRTrainer(cfg)

    # Optionally resume
    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        trainer.model = MIRModel.from_pretrained(args.resume, device=device)
        trainer.model = trainer.model.to(trainer.device)

    # ---- Train ----
    logger.info("Starting training...")
    history = trainer.fit(train_loader, val_loader)

    # ---- Summary ----
    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info("Best val loss   : %.4f", trainer.early_stopping.best)
    logger.info("Epochs trained  : %d", len(history["train_loss"]))
    logger.info("Final train loss: %.4f", history["train_loss"][-1])
    logger.info("Checkpoint saved to: %s/encoder_best.pth", args.output)


if __name__ == "__main__":
    main()
