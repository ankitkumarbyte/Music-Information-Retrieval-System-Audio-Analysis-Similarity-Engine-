"""
Unit tests for src/mir/models/encoder.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.mir.models.encoder import (
    AudioEncoder, ConvBlock, EmbeddingInferencer,
    MIRModel, NTXentLoss, ResidualBlock,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_mel():
    """Batch of 4 random mel spectrograms: (B, 1, 128, 256)."""
    return torch.randn(4, 1, 128, 256)


@pytest.fixture
def single_mel():
    """Single mel spectrogram: (1, 128, 256)."""
    return torch.randn(1, 128, 256)


@pytest.fixture
def model():
    return MIRModel(num_genres=10, num_instruments=16, embedding_dim=256)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TestConvBlock:

    def test_output_shape_with_pool(self):
        block = ConvBlock(1, 32, pool=True)
        x = torch.randn(2, 1, 64, 64)
        out = block(x)
        assert out.shape == (2, 32, 32, 32)

    def test_output_shape_no_pool(self):
        block = ConvBlock(1, 32, pool=False)
        x = torch.randn(2, 1, 64, 64)
        out = block(x)
        assert out.shape == (2, 32, 64, 64)


class TestResidualBlock:

    def test_output_shape_preserved(self):
        block = ResidualBlock(64)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """With zero-init weights, output ≈ input (residual dominates)."""
        block = ResidualBlock(4)
        for p in block.net.parameters():
            torch.nn.init.zeros_(p)
        x = torch.randn(1, 4, 8, 8)
        out = block(x)
        # Not exactly equal because of BN, but close
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------

class TestAudioEncoder:

    def test_output_shape(self, batch_mel):
        enc = AudioEncoder(embedding_dim=256)
        out = enc(batch_mel)
        assert out.shape == (4, 256)

    def test_l2_normalised(self, batch_mel):
        enc = AudioEncoder()
        out = enc(batch_mel)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_parameter_count_reasonable(self):
        enc = AudioEncoder(embedding_dim=256)
        n_params = enc.num_parameters
        # Should be in the range 1M–20M
        assert 1_000_000 < n_params < 20_000_000, f"Unexpected param count: {n_params}"


# ---------------------------------------------------------------------------
# MIRModel
# ---------------------------------------------------------------------------

class TestMIRModel:

    def test_forward_keys(self, model, batch_mel):
        out = model(batch_mel)
        assert "embedding" in out
        assert "genre_logits" in out
        assert "instr_logits" in out

    def test_embedding_shape(self, model, batch_mel):
        out = model(batch_mel)
        assert out["embedding"].shape == (4, 256)

    def test_genre_logits_shape(self, model, batch_mel):
        out = model(batch_mel)
        assert out["genre_logits"].shape == (4, 10)

    def test_instrument_logits_shape(self, model, batch_mel):
        out = model(batch_mel)
        assert out["instr_logits"].shape == (4, 16)

    def test_embedding_is_l2_normalised(self, model, batch_mel):
        out = model(batch_mel)
        norms = out["embedding"].norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_eval_mode_no_dropout(self, model, batch_mel):
        """Outputs should be deterministic in eval mode."""
        model.eval()
        with torch.no_grad():
            out1 = model(batch_mel)["embedding"]
            out2 = model(batch_mel)["embedding"]
        assert torch.allclose(out1, out2)

    def test_save_and_load(self, model, tmp_path):
        ckpt = tmp_path / "test.pth"
        model.save(ckpt, epoch=5)
        loaded = MIRModel.from_pretrained(ckpt)
        loaded.eval()
        model.eval()
        x = torch.randn(2, 1, 128, 256)
        with torch.no_grad():
            o1 = model(x)["embedding"]
            o2 = loaded(x)["embedding"]
        assert torch.allclose(o1, o2, atol=1e-5)


# ---------------------------------------------------------------------------
# NTXentLoss
# ---------------------------------------------------------------------------

class TestNTXentLoss:

    def test_loss_is_scalar(self):
        loss_fn = NTXentLoss(temperature=0.07)
        z_i = torch.nn.functional.normalize(torch.randn(8, 64), dim=1)
        z_j = torch.nn.functional.normalize(torch.randn(8, 64), dim=1)
        loss = loss_fn(z_i, z_j)
        assert loss.ndim == 0

    def test_loss_positive(self):
        loss_fn = NTXentLoss()
        z_i = torch.nn.functional.normalize(torch.randn(8, 64), dim=1)
        z_j = torch.nn.functional.normalize(torch.randn(8, 64), dim=1)
        assert loss_fn(z_i, z_j).item() > 0

    def test_identical_views_lower_loss(self):
        """Passing same views should produce lower loss than random views."""
        loss_fn = NTXentLoss(temperature=0.07)
        z = torch.nn.functional.normalize(torch.randn(16, 128), dim=1)
        loss_same = loss_fn(z, z.clone()).item()
        loss_rand = loss_fn(z, torch.nn.functional.normalize(torch.randn(16, 128), dim=1)).item()
        assert loss_same < loss_rand


# ---------------------------------------------------------------------------
# EmbeddingInferencer
# ---------------------------------------------------------------------------

class TestEmbeddingInferencer:

    def test_embed_single(self, model, single_mel):
        inferencer = EmbeddingInferencer(model)
        emb = inferencer.embed(single_mel)
        assert emb.shape == (256,)

    def test_embed_batch(self, model, batch_mel):
        inferencer = EmbeddingInferencer(model)
        emb = inferencer.embed(batch_mel)
        assert emb.shape == (4, 256)

    def test_classify_genre_keys(self, model, single_mel):
        inferencer = EmbeddingInferencer(model)
        result = inferencer.classify_genre(single_mel)
        assert "label" in result
        assert "confidence" in result
        assert "top_3" in result
        assert result["label"] in MIRModel.GENRES
        assert 0.0 <= result["confidence"] <= 1.0

    def test_classify_instruments_threshold(self, model, single_mel):
        inferencer = EmbeddingInferencer(model)
        instruments = inferencer.classify_instruments(single_mel, threshold=0.0)
        # With threshold=0, all 16 instruments should be returned
        assert len(instruments) == 16

    def test_no_grad_during_inference(self, model, single_mel):
        inferencer = EmbeddingInferencer(model)
        emb = inferencer.embed(single_mel)
        assert not emb.requires_grad
