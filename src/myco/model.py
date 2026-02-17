"""Model definition for MoCo v3 training on nuclei crops.

Schedules follow the official MoCo v3 reference implementation:
https://github.com/facebookresearch/moco-v3
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import math
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class EmbeddingEncoder(ABC):
    """Abstract interface for encoder modules that produce embeddings."""

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings for input batch ``x``."""


class MLPHead(nn.Module):
    """Configurable MLP head used for projector/predictor blocks.

    Parameters
    ----------
    in_dim : int
        Input embedding dimension.
    hidden : int
        Hidden dimension for intermediate layers.
    out_dim : int
        Output embedding dimension.
    num_layers : int
        Number of linear layers in the MLP.
    last_bn : bool
        Whether to append batch norm to the final layer output.
    last_bn_affine : bool
        Whether the final batch norm has affine parameters.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        num_layers: int,
        last_bn: bool = True,
        last_bn_affine: bool = False,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1."
        layers: list[nn.Module] = []
        for layer_idx in range(num_layers):
            dim1 = in_dim if layer_idx == 0 else hidden
            dim2 = out_dim if layer_idx == (num_layers - 1) else hidden
            layers.append(nn.Linear(dim1, dim2, bias=False))
            if layer_idx < (num_layers - 1):
                layers.append(nn.BatchNorm1d(dim2))
                layers.append(nn.ReLU(inplace=True))
            elif last_bn:
                layers.append(nn.BatchNorm1d(dim2, affine=last_bn_affine))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        img_size: int = 40,
        big_size: int = 60,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["init_ckpt"])

        self.temperature = temperature
        self.base_m = base_m
        self.total_epochs = epochs
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.warmup_epochs = max(0, warmup_epochs)
        self.total_steps = self.steps_per_epoch * self.total_epochs
        self.cells_sampled = 0

        assert img_size > 0, "img_size must be positive."
        assert big_size > 0, "big_size must be positive."
        self.img_size = img_size
        self.big_size = big_size

        self.q_enc = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
            img_size=img_size,
        )
        dim = self.q_enc.num_features
        self.q_proj = MLPHead(
            dim, mlp_hidden, proj_dim, num_layers=3, last_bn=True, last_bn_affine=False
        )

        self.k_enc = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
            img_size=img_size,
        )
        self.k_proj = MLPHead(
            dim, mlp_hidden, proj_dim, num_layers=3, last_bn=True, last_bn_affine=False
        )
        self.predictor = MLPHead(
            proj_dim,
            mlp_hidden,
            proj_dim,
            num_layers=2,
            last_bn=True,
            last_bn_affine=True,
        )

        self._copy_q_to_k()

        if init_ckpt and os.path.isfile(init_ckpt):
            ckpt = torch.load(init_ckpt, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            missing, unexpected = self.load_state_dict(state, strict=False)
            logger.info(
                "[init] loaded %s missing=%d unexpected=%d",
                init_ckpt,
                len(missing),
                len(unexpected),
            )

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
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

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
        return float(self.current_epoch) + (
            float(batch_idx) / float(self.steps_per_epoch)
        )

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

    @torch.no_grad()
    def _concat_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors across ranks without gradients.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor of shape ``(N, D)``.

        Returns
        -------
        torch.Tensor
            Concatenated tensor of shape ``(N * world_size, D)``.
        """
        assert tensor.ndim == 2, f"Expected 2D tensor, got {tuple(tensor.shape)}."
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return tensor
        world_size = torch.distributed.get_world_size()
        tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)

    def _contrastive_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """MoCo v3 contrastive loss with all-gathered keys and rank-aware labels."""
        assert q.ndim == 2 and k.ndim == 2, "Expected 2D embeddings."
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        gathered_k = self._concat_all_gather(k)
        logits = torch.einsum("nc,mc->nm", q, gathered_k) / self.temperature
        local_batch = q.shape[0]
        rank_offset = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_offset = local_batch * torch.distributed.get_rank()
        labels = (
            torch.arange(local_batch, dtype=torch.long, device=logits.device)
            + rank_offset
        )
        loss = F.cross_entropy(logits, labels)
        return loss * (2.0 * self.temperature)

    def training_step(self, batch, batch_idx):
        """Run one contrastive training step."""
        assert isinstance(batch, (list, tuple)), (
            "Expected batch to be a tuple/list of tensors."
        )
        assert len(batch) == 2, "Expected batch to contain exactly two tensors."
        x1, x2 = batch
        assert isinstance(x1, torch.Tensor), "Expected x1 to be a torch.Tensor."
        assert isinstance(x2, torch.Tensor), "Expected x2 to be a torch.Tensor."
        assert x1.ndim == 4, (
            "Expected x1 to have shape (batch, channels, height, width)."
        )
        assert x2.ndim == 4, (
            "Expected x2 to have shape (batch, channels, height, width)."
        )
        assert x1.shape == x2.shape, "Expected x1 and x2 to share identical shapes."
        assert x1.shape[0] > 0, "Expected a positive batch size."
        assert x1.shape[-2:] == (self.img_size, self.img_size), (
            f"Expected x1 spatial size {(self.img_size, self.img_size)}, got {x1.shape[-2:]}."
        )
        self._update_learning_rate(batch_idx)
        momentum = self._momentum_schedule(self._epoch_progress(batch_idx))
        with torch.no_grad():
            self._momentum_update(momentum)

        q1 = self.predictor(self.q_proj(self.q_enc(x1)))
        q2 = self.predictor(self.q_proj(self.q_enc(x2)))
        with torch.no_grad():
            k1 = self.k_proj(self.k_enc(x1))
            k2 = self.k_proj(self.k_enc(x2))

        loss = self._contrastive_loss(q1, k2) + self._contrastive_loss(q2, k1)
        self.cells_sampled += int(x1.shape[0])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "cells_sampled",
            float(self.cells_sampled),
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=False,
        )
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
