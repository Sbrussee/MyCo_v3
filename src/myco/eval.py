"""Evaluation callback for representation metrics and probe training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torchvision.transforms import ToTensor

from .augment import RotationCrop40
from .data import SlideEntry, load_centroids, safe_open_slide
from .model import MoCoV3Lit
from .utils import read_patch
from .visualization import MosaicConfig, create_patch_mosaic

logger = logging.getLogger(__name__)


class AttentionPool(nn.Module):
    """Attention pooling for slide-level classification."""

    def __init__(self, in_dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.v_proj = nn.Linear(in_dim, attn_dim)
        self.u_proj = nn.Linear(in_dim, attn_dim)
        self.w_proj = nn.Linear(attn_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weights = torch.tanh(self.v_proj(hidden)) * torch.sigmoid(self.u_proj(hidden))
        weights = self.w_proj(weights).squeeze(1)
        weights = torch.softmax(weights, dim=0)
        return torch.sum(weights.unsqueeze(1) * hidden, dim=0)


@dataclass(frozen=True)
class ProbeConfig:
    """Configuration for linear-probe evaluation."""

    cells_per_slide: int = 200
    probe_epochs: int = 20
    probe_lr: float = 1e-3
    slides_per_class: int = 150
    seed: int = 0


@dataclass(frozen=True)
class BestCheckpointState:
    """Track the best probe metric for checkpointing."""

    metric_name: str
    metric_value: float
    epoch: int
    path: str


def repr_metrics_np(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute simple representation statistics from embeddings.

    Expected input shape: (n_samples, embedding_dim).
    """
    assert embeddings.ndim == 2, (
        f"Expected 2D embeddings array, got shape {embeddings.shape}."
    )
    if embeddings.size == 0:
        logger.warning("No embeddings available for representation metrics.")
        return {"embedding_variance": float("nan"), "mean_embedding_norm": float("nan")}

    finite_rows = np.isfinite(embeddings).all(axis=1)
    if not np.all(finite_rows):
        logger.warning(
            "Detected %d/%d embeddings with non-finite values; filtering before metrics.",
            int((~finite_rows).sum()),
            embeddings.shape[0],
        )
    embeddings = embeddings[finite_rows]
    if embeddings.size == 0:
        logger.error(
            "All embeddings were non-finite; representation metrics are undefined."
        )
        return {"embedding_variance": float("nan"), "mean_embedding_norm": float("nan")}

    return {
        "embedding_variance": float(np.mean(np.var(embeddings, axis=0))),
        "mean_embedding_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
    }


class EvalCallback(pl.Callback):
    """Runs slide-level probe and patch mosaics after each epoch."""

    def __init__(
        self,
        entries: List[SlideEntry],
        slide_labels: Dict[str, int],
        output_dir: str,
        probe: ProbeConfig,
        mosaic: MosaicConfig,
    ) -> None:
        super().__init__()
        self.entries = entries
        self.slide_labels = slide_labels
        self.output_dir = output_dir
        self.probe = probe
        self.mosaic = mosaic

        self.rotcrop = RotationCrop40(
            big_size=60,
            out_size=40,
            degrees=360.0,
        )
        self.totensor = ToTensor()

        self._probe_subset: Optional[List[SlideEntry]] = None
        self._best_state: Optional[BestCheckpointState] = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: MoCoV3Lit) -> None:
        if trainer.global_rank != 0:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self.rotcrop = RotationCrop40(
            big_size=pl_module.big_size,
            out_size=pl_module.img_size,
            degrees=360.0,
        )
        self._probe_subset = self._select_probe_subset()
        subset_path = os.path.join(self.output_dir, "probe_subset.json")
        with open(subset_path, "w", encoding="utf-8") as handle:
            json.dump(
                [entry.slide_id for entry in self._probe_subset], handle, indent=2
            )
        num_labels = len(self.slide_labels)
        positives = sum(1 for label in self.slide_labels.values() if label == 1)
        negatives = sum(1 for label in self.slide_labels.values() if label == 0)
        logger.info(
            "Eval probe setup: %d entries, %d labels (%d positive, %d negative).",
            len(self.entries),
            num_labels,
            positives,
            negatives,
        )
        logger.info(
            "Probe subset size=%d cells_per_slide=%d probe_epochs=%d.",
            len(self._probe_subset),
            self.probe.cells_per_slide,
            self.probe.probe_epochs,
        )

    def _select_probe_subset(self) -> List[SlideEntry]:
        rng = random.Random(self.probe.seed)
        positives = [e for e in self.entries if self.slide_labels.get(e.slide_id) == 1]
        negatives = [e for e in self.entries if self.slide_labels.get(e.slide_id) == 0]
        if not positives:
            logger.warning("No positive slides available for probe evaluation.")
        if not negatives:
            logger.warning("No negative slides available for probe evaluation.")
        rng.shuffle(positives)
        rng.shuffle(negatives)
        selected = (
            positives[: self.probe.slides_per_class]
            + negatives[: self.probe.slides_per_class]
        )
        logger.info(
            "Selected %d/%d positives and %d/%d negatives for probe subset.",
            min(len(positives), self.probe.slides_per_class),
            len(positives),
            min(len(negatives), self.probe.slides_per_class),
            len(negatives),
        )
        return selected

    @torch.no_grad()
    def _embed_slide(
        self,
        pl_module: MoCoV3Lit,
        device: torch.device,
        entry: SlideEntry,
        centroids: List[Tuple[float, float]],
        rng: random.Random,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        assert centroids, f"Expected non-empty centroids for slide {entry.slide_id}."
        embs: List[np.ndarray] = []
        patches: List[np.ndarray] = []
        slide = safe_open_slide(entry.wsi_path)
        try:
            for _ in range(self.probe.cells_per_slide):
                center = rng.choice(centroids)
                patch = read_patch(slide, center, 60)
                patch = self.rotcrop(patch)
                x = self.totensor(patch).unsqueeze(0).to(device)
                z = pl_module.encode(x).squeeze(0).detach().cpu().numpy()
                if not np.isfinite(z).all():
                    logger.warning(
                        "Non-finite embedding for slide %s; skipping this patch.",
                        entry.slide_id,
                    )
                    continue
                embs.append(z)
                patches.append(np.array(patch))
        finally:
            slide.close()
        if not embs:
            logger.warning(
                "No valid embeddings extracted for slide %s (centroids=%d).",
                entry.slide_id,
                len(centroids),
            )
            return np.empty((0, pl_module.hparams.proj_dim), dtype=np.float32), []
        return np.stack(embs, axis=0), patches

    def _collect_embeddings(
        self, pl_module: MoCoV3Lit, device: torch.device, rng: random.Random
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[np.ndarray], np.ndarray]:
        embeds_by_slide: Dict[str, np.ndarray] = {}
        labels: Dict[str, int] = {}
        mosaic_patches: List[np.ndarray] = []
        mosaic_embeddings: List[np.ndarray] = []
        skipped_missing_label = 0
        skipped_missing_centroids = 0
        skipped_empty_embeddings = 0

        if self._probe_subset is None:
            self._probe_subset = self._select_probe_subset()

        for entry in self._probe_subset:
            label = self.slide_labels.get(entry.slide_id)
            if label is None:
                skipped_missing_label += 1
                continue
            centroids = load_centroids(entry.ann_path, slide_path=entry.wsi_path)
            if not centroids:
                skipped_missing_centroids += 1
                continue
            embeddings, patches = self._embed_slide(
                pl_module, device, entry, centroids, rng
            )
            if embeddings.shape[0] == 0:
                skipped_empty_embeddings += 1
                continue
            embeds_by_slide[entry.slide_id] = embeddings
            labels[entry.slide_id] = int(label)
            mosaic_embeddings.append(embeddings)
            mosaic_patches.extend(patches)

        if mosaic_embeddings:
            mosaic_array = np.concatenate(mosaic_embeddings, axis=0)
        else:
            mosaic_array = np.empty((0, pl_module.hparams.proj_dim))

        if mosaic_array.shape[0] > self.mosaic.max_points:
            indices = rng.sample(range(mosaic_array.shape[0]), self.mosaic.max_points)
            mosaic_array = mosaic_array[indices]
            mosaic_patches = [mosaic_patches[i] for i in indices]

        logger.info(
            "Collected embeddings for %d slides (skipped: %d missing labels, %d missing centroids, %d empty embeddings).",
            len(embeds_by_slide),
            skipped_missing_label,
            skipped_missing_centroids,
            skipped_empty_embeddings,
        )
        return embeds_by_slide, labels, mosaic_patches, mosaic_array

    def _train_probe(
        self,
        embeds: Dict[str, np.ndarray],
        labels: Dict[str, int],
        device: torch.device,
    ) -> Dict[str, float]:
        keys = list(embeds.keys())
        if len(keys) < 4:
            logger.warning(
                "Not enough slides (%d) for probe training; skipping.", len(keys)
            )
            return {"probe_auc": float("nan"), "probe_bal_acc": float("nan")}

        rng = np.random.default_rng(self.probe.seed)
        rng.shuffle(keys)
        split = int(0.8 * len(keys))
        train_keys, val_keys = keys[:split], keys[split:]
        if not val_keys:
            logger.warning("Probe validation split is empty; skipping metrics.")
            return {"probe_auc": float("nan"), "probe_bal_acc": float("nan")}

        embedding_dim = embeds[keys[0]].shape[1]
        assert embedding_dim > 0, "Embedding dimension must be positive."
        attn = AttentionPool(embedding_dim, 128).to(device)
        clf = nn.Linear(embedding_dim, 1).to(device)
        opt = torch.optim.AdamW(
            list(attn.parameters()) + list(clf.parameters()),
            lr=self.probe.probe_lr,
            weight_decay=1e-4,
        )

        for _ in range(self.probe.probe_epochs):
            attn.train()
            clf.train()
            for slide_id in train_keys:
                hidden = torch.from_numpy(embeds[slide_id]).float().to(device)
                assert hidden.ndim == 2, (
                    f"Expected embeddings to be 2D, got {hidden.shape}."
                )
                target = torch.tensor(
                    [labels[slide_id]], dtype=torch.float32, device=device
                )
                pooled = attn(hidden)
                logit = clf(pooled).squeeze(0)
                loss = F.binary_cross_entropy_with_logits(logit, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        attn.eval()
        clf.eval()
        y_true: List[float] = []
        y_score: List[float] = []
        with torch.no_grad():
            for slide_id in val_keys:
                hidden = torch.from_numpy(embeds[slide_id]).float().to(device)
                label = float(labels[slide_id])
                pooled = attn(hidden)
                prob = torch.sigmoid(clf(pooled)).item()
                y_true.append(label)
                y_score.append(prob)

        metrics: Dict[str, float] = {}
        if len(set(y_true)) > 1:
            metrics["probe_auc"] = float(roc_auc_score(y_true, y_score))
        else:
            metrics["probe_auc"] = float("nan")
        y_pred = [1 if score >= 0.5 else 0 for score in y_score]
        metrics["probe_bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        return metrics

    def _maybe_save_best(
        self, pl_module: MoCoV3Lit, epoch: int, metrics: Dict[str, float]
    ) -> Optional[BestCheckpointState]:
        metric_name = "probe_bal_acc"
        metric_value = metrics.get(metric_name, float("nan"))
        if np.isnan(metric_value):
            return None

        if self._best_state is None or metric_value > self._best_state.metric_value:
            path = os.path.join(self.output_dir, "best_probe_weights.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "metric": metric_value,
                    "metric_name": metric_name,
                    "weights": pl_module.extract_weights(),
                },
                path,
            )
            self._best_state = BestCheckpointState(
                metric_name=metric_name,
                metric_value=metric_value,
                epoch=epoch,
                path=path,
            )
        return self._best_state

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: MoCoV3Lit) -> None:
        if trainer.global_rank != 0:
            return

        device = pl_module.device
        rng = random.Random(self.probe.seed + trainer.current_epoch)

        dl = trainer.train_dataloader
        if isinstance(dl, list):
            dl = dl[0]
        batches: List[np.ndarray] = []
        iterator = iter(dl)
        for _ in range(10):
            try:
                x1, _ = next(iterator)
            except StopIteration:
                break
            x1 = x1.to(device)
            z = pl_module.encode(x1).detach().cpu().numpy()
            batches.append(z)
        if batches:
            embeddings_np = np.concatenate(batches, axis=0)
            repr_metrics = repr_metrics_np(embeddings_np)
        else:
            logger.warning(
                "No training batches available for representation metrics at epoch %d.",
                trainer.current_epoch,
            )
            repr_metrics = {
                "embedding_variance": float("nan"),
                "mean_embedding_norm": float("nan"),
            }

        embeds_by_slide, labels, mosaic_patches, mosaic_embeddings = (
            self._collect_embeddings(pl_module, device, rng)
        )
        probe_metrics = self._train_probe(embeds_by_slide, labels, device=device)
        best_state = self._maybe_save_best(
            pl_module, trainer.current_epoch, probe_metrics
        )

        metrics = {**repr_metrics, **probe_metrics}
        trainer.logger.log_metrics(metrics, step=trainer.global_step)
        if best_state is not None:
            trainer.logger.log_metrics(
                {"best_probe_bal_acc": best_state.metric_value},
                step=trainer.global_step,
            )

        if mosaic_embeddings.shape[0] > 0:
            mosaic_path = os.path.join(
                self.output_dir, f"mosaic_epoch_{trainer.current_epoch:03d}.png"
            )
            title = f"Epoch {trainer.current_epoch} {self.mosaic.method.upper()} Mosaic"
            create_patch_mosaic(
                mosaic_embeddings,
                mosaic_patches,
                self.mosaic,
                title=title,
                output_path=mosaic_path,
            )

        print(
            f"[eval@epoch={trainer.current_epoch}] "
            f"var={repr_metrics['embedding_variance']:.6f} "
            f"norm={repr_metrics['mean_embedding_norm']:.3f} "
            f"auc={probe_metrics.get('probe_auc', float('nan')):.3f} "
            f"bal_acc={probe_metrics.get('probe_bal_acc', float('nan')):.3f}"
        )
