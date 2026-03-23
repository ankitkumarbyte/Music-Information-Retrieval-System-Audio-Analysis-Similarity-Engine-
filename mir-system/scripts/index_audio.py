#!/usr/bin/env python3
"""
Bulk Audio Indexer
==================
Preprocess and index a directory of audio files into the FAISS search engine.

Usage
-----
    python scripts/index_audio.py \\
        --audio_dir  /path/to/music/ \\
        --index_out  data/faiss/index.faiss \\
        --model      checkpoints/encoder_best.pth \\
        --batch      32 \\
        --device     cpu

Output
------
Saves a FAISS index + ID map to --index_out.
Prints a summary table of processed / failed files.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from src.mir.features.spectrogram import compute_mel_spectrogram, mel_to_tensor
from src.mir.models.encoder import EmbeddingInferencer, MIRModel
from src.mir.preprocessing.pipeline import AudioPreprocessingPipeline, PipelineConfig
from src.mir.search.faiss_index import FaissSearchEngine
from src.mir.utils.audio_io import SUPPORTED_EXTENSIONS, load_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("indexer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk audio indexer")
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--index_out", default="data/faiss/index.faiss")
    p.add_argument("--model", default=None, help="Path to model checkpoint (.pth)")
    p.add_argument("--batch", type=int, default=32, help="Embedding batch size")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--nlist", type=int, default=256, help="FAISS nlist parameter")
    p.add_argument("--resume", default=None, help="Existing FAISS index to extend")
    return p.parse_args()


def discover_files(audio_dir: Path, max_files: int | None) -> list[Path]:
    """Recursively find all supported audio files."""
    files = [
        p for p in sorted(audio_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    logger.info("Found %d audio files in %s", len(files), audio_dir)
    if max_files:
        files = files[:max_files]
        logger.info("Capped at %d files (--max_files)", max_files)
    return files


def embed_batch(
    files: list[Path],
    pipeline: AudioPreprocessingPipeline,
    inferencer: EmbeddingInferencer,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Process a batch of files → embeddings.

    Returns
    -------
    embeddings : np.ndarray  shape (N_success, 256)
    track_ids  : list[str]
    failed     : list[str]  filenames that errored
    """
    embeddings = []
    track_ids = []
    failed = []

    for fpath in files:
        try:
            y, sr = pipeline.process_file(fpath)
            mel = compute_mel_spectrogram(y, sr)
            tensor = mel_to_tensor(mel)
            emb = inferencer.embed(tensor)
            embeddings.append(emb.numpy())
            track_ids.append(fpath.stem)   # use filename stem as ID
        except Exception as e:
            logger.warning("Failed to process %s: %s", fpath.name, e)
            failed.append(fpath.name)

    if embeddings:
        return np.array(embeddings, dtype=np.float32), track_ids, failed
    return np.empty((0, 256), dtype=np.float32), [], failed


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        logger.error("Not a directory: %s", audio_dir)
        sys.exit(1)

    # ---- Components ----
    pipeline = AudioPreprocessingPipeline(PipelineConfig())

    if args.model and Path(args.model).exists():
        model = MIRModel.from_pretrained(args.model, device=args.device)
    else:
        logger.warning("No checkpoint found — using untrained model (embeddings won't be meaningful).")
        model = MIRModel()

    inferencer = EmbeddingInferencer(model, device=args.device)

    # ---- FAISS index ----
    if args.resume and Path(args.resume).exists():
        engine = FaissSearchEngine.load(args.resume)
        logger.info("Resumed index with %d vectors", len(engine))
    else:
        engine = FaissSearchEngine(dim=256, nlist=args.nlist, M=0, nprobe=32)

    # ---- Discover files ----
    files = discover_files(audio_dir, args.max_files)
    if not files:
        logger.error("No audio files found.")
        sys.exit(1)

    # ---- Train index on first batch if needed ----
    if not engine._index.is_trained:
        logger.info("Training FAISS index...")
        train_files = files[:min(len(files), max(args.nlist * 39, 500))]
        train_embs, _, _ = embed_batch(train_files, pipeline, inferencer)
        if len(train_embs) < args.nlist:
            # Not enough vectors for IVF — fall back to flat index
            logger.warning(
                "Only %d training vectors for nlist=%d; switching to flat index.",
                len(train_embs), args.nlist,
            )
            engine = FaissSearchEngine(dim=256, nlist=1, M=0, nprobe=1)
            dummy = np.random.randn(50, 256).astype(np.float32)
            dummy /= np.linalg.norm(dummy, axis=1, keepdims=True)
            engine.train(dummy)
        else:
            engine.train(train_embs)

    # ---- Index all files in batches ----
    total_ok = 0
    total_fail = 0

    for i in tqdm(range(0, len(files), args.batch), desc="Indexing", unit="batch"):
        batch_files = files[i: i + args.batch]
        embs, ids, fails = embed_batch(batch_files, pipeline, inferencer)
        if len(embs):
            engine.add(embs, ids)
            total_ok += len(ids)
        total_fail += len(fails)

    # ---- Save ----
    out_path = Path(args.index_out)
    engine.save(out_path)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 55)
    logger.info("Indexing complete in %.1fs", elapsed)
    logger.info("  Indexed successfully : %d", total_ok)
    logger.info("  Failed               : %d", total_fail)
    logger.info("  Total in index       : %d", len(engine))
    logger.info("  Index saved to       : %s", out_path)
    logger.info("  Throughput           : %.1f files/s", total_ok / elapsed if elapsed > 0 else 0)


if __name__ == "__main__":
    main()
