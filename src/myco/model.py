"""Model definition for MoCo v3 training on nuclei crops.

Schedules follow the official MoCo v3 reference implementation:
https://github.com/facebookresearch/moco-v3
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import math
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pytorch_lightning as pl


class EmbeddingEncoder(ABC):
    """Abstract interface for encoder modules that produce embeddings."""

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings for input batch ``x``."""


class MLPHead(nn.Module):
    """Two-layer MLP projection head used by MoCo v3."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x


class MoCoV3Lit(pl.LightningModule, EmbeddingEncoder):
    """Lightning module implementing MoCo v3 for nuclei embeddings."""

    def __init__(
        self,
        init_ckpt: str,
        lr: float,
        weight_decay: float,
        temperature: float,
        proj_dim: int,
        mlp_hidden: int,
        base_m: float,
        epochs: int,
        steps_per_epoch: int,
        warmup_epochs: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["init_ckpt"])

        self.temperature = temperature
        self.base_m = base_m
        self.total_epochs = epochs
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.warmup_epochs = max(0, warmup_epochs)
        self.total_steps = self.steps_per_epoch * self.total_epochs

        self.q_enc = timm.create_model("vit_small_patch8_224", pretrained=False, num_classes=0, global_pool="avg")
        dim = self.q_enc.num_features
        self.q_proj = MLPHead(dim, mlp_hidden, proj_dim)

        self.k_enc = timm.create_model("vit_small_patch8_224", pretrained=False, num_classes=0, global_pool="avg")
        self.k_proj = MLPHead(dim, mlp_hidden, proj_dim)

        self._copy_q_to_k()

        if init_ckpt and os.path.isfile(init_ckpt):
            ckpt = torch.load(init_ckpt, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            missing, unexpected = self.load_state_dict(state, strict=False)
            print(f"[init] loaded {init_ckpt} missing={len(missing)} unexpected={len(unexpected)}")

    @torch.no_grad()
    def _copy_q_to_k(self) -> None:
        for q_param, k_param in zip(self.q_enc.parameters(), self.k_enc.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad = False
        for q_param, k_param in zip(self.q_proj.parameters(), self.k_proj.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self, momentum: float) -> None:
        for q_param, k_param in zip(self.q_enc.parameters(), self.k_enc.parameters()):
            k_param.data = k_param.data * momentum + q_param.data * (1.0 - momentum)
        for q_param, k_param in zip(self.q_proj.parameters(), self.k_proj.parameters()):
            k_param.data = k_param.data * momentum + q_param.data * (1.0 - momentum)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def _lr_schedule(self, epoch_progress: float) -> float:
        """Cosine learning-rate schedule with linear warmup (MoCo v3)."""
        if self.total_epochs <= 1:
            return 1.0
        if self.warmup_epochs > 0 and epoch_progress < self.warmup_epochs:
            return epoch_progress / float(self.warmup_epochs)
        cosine_progress = epoch_progress - self.warmup_epochs
        cosine_total = max(1e-12, float(self.total_epochs - self.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * cosine_progress / cosine_total))

    def _momentum_schedule(self, epoch_progress: float) -> float:
        """Cosine momentum schedule aligned with the MoCo v3 reference implementation."""
        if self.total_epochs <= 1:
            return self.base_m
        progress = epoch_progress / float(self.total_epochs)
        return 1.0 - (1.0 - self.base_m) * (0.5 * (1.0 + math.cos(math.pi * progress)))

    def _epoch_progress(self, batch_idx: int) -> float:
        """Return the fractional epoch progress for a batch index."""
        return float(self.current_epoch) + (float(batch_idx) / float(self.steps_per_epoch))

    def _update_learning_rate(self, batch_idx: int) -> None:
        """Update optimizer learning rates to match the MoCo v3 schedule."""
        optimizer = self.optimizers()
        if optimizer is None:
            return
        if isinstance(optimizer, (list, tuple)):
            if not optimizer:
                return
            optimizer = optimizer[0]
        lr_scale = self._lr_schedule(self._epoch_progress(batch_idx))
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.hparams.lr * lr_scale

    def training_step(self, batch, batch_idx):
        """Run one contrastive training step."""
        assert isinstance(batch, (list, tuple)), "Expected batch to be a tuple/list of tensors."
        assert len(batch) == 2, "Expected batch to contain exactly two tensors."
        x1, x2 = batch
        assert isinstance(x1, torch.Tensor), "Expected x1 to be a torch.Tensor."
        assert isinstance(x2, torch.Tensor), "Expected x2 to be a torch.Tensor."
        assert x1.ndim == 4, "Expected x1 to have shape (batch, channels, height, width)."
        assert x2.ndim == 4, "Expected x2 to have shape (batch, channels, height, width)."
        assert x1.shape == x2.shape, "Expected x1 and x2 to share identical shapes."
        assert x1.shape[0] > 0, "Expected a positive batch size."
        self._update_learning_rate(batch_idx)
        momentum = self._momentum_schedule(self._epoch_progress(batch_idx))
        with torch.no_grad():
            self._momentum_update(momentum)

        q1 = F.normalize(self.q_proj(self.q_enc(x1)), dim=1)
        q2 = F.normalize(self.q_proj(self.q_enc(x2)), dim=1)
        with torch.no_grad():
            k1 = F.normalize(self.k_proj(self.k_enc(x1)), dim=1)
            k2 = F.normalize(self.k_proj(self.k_enc(x2)), dim=1)

        logits_12 = (q1 @ k2.t()) / self.temperature
        logits_21 = (q2 @ k1.t()) / self.temperature
        labels = torch.arange(logits_12.size(0), device=logits_12.device)

        loss = 0.5 * (F.cross_entropy(logits_12, labels) + F.cross_entropy(logits_21, labels))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images into projection embeddings."""
        assert isinstance(x, torch.Tensor), "Expected x to be a torch.Tensor."
        assert x.ndim == 4, "Expected x to have shape (batch, channels, height, width)."
        assert x.shape[0] > 0, "Expected a positive batch size."
        return self.q_proj(self.q_enc(x))

    def extract_weights(self) -> Dict[str, torch.Tensor]:
        """Return a minimal state dict for inference."""
        return {
            "q_enc": self.q_enc.state_dict(),
            "q_proj": self.q_proj.state_dict(),
            "hyperparameters": dict(self.hparams),
        }
