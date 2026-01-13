"""Model definition for MoCo v3 training on nuclei crops."""
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
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["init_ckpt"])

        self.temperature = temperature
        self.base_m = base_m
        self.total_epochs = epochs

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

    def _momentum_schedule(self) -> float:
        progress = self.current_epoch / max(1, (self.total_epochs - 1))
        return 1.0 - (1.0 - self.base_m) * (0.5 * (1.0 + math.cos(math.pi * progress)))

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        momentum = self._momentum_schedule()
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
        return self.q_proj(self.q_enc(x))

    def extract_weights(self) -> Dict[str, torch.Tensor]:
        """Return a minimal state dict for inference."""
        return {
            "q_enc": self.q_enc.state_dict(),
            "q_proj": self.q_proj.state_dict(),
            "hyperparameters": dict(self.hparams),
        }
