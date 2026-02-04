"""Training callbacks for logging metrics and progress."""

from __future__ import annotations

import resource
import time
from typing import Sequence

import pytorch_lightning as pl
import torch


def extract_batch_size(batch: Sequence[torch.Tensor]) -> int:
    """Return the batch size from a contrastive pair batch.

    Expected batch structure: ``(x1, x2)`` where each tensor is shaped
    ``(batch, channels, height, width)`` and both tensors match in shape.
    """
    assert isinstance(batch, (list, tuple)), (
        "Expected batch to be a tuple/list of tensors."
    )
    assert len(batch) == 2, "Expected batch to contain exactly two tensors."
    x1, x2 = batch
    assert isinstance(x1, torch.Tensor), (
        "Expected the first batch element to be a torch.Tensor."
    )
    assert isinstance(x2, torch.Tensor), (
        "Expected the second batch element to be a torch.Tensor."
    )
    assert x1.ndim == 4, "Expected x1 to have shape (batch, channels, height, width)."
    assert x2.ndim == 4, "Expected x2 to have shape (batch, channels, height, width)."
    assert x1.shape == x2.shape, "Expected x1 and x2 to share identical shapes."
    return int(x1.shape[0])


class BatchMetricsLogger(pl.Callback):
    """Log throughput and memory metrics every ``log_every_n_batches`` steps."""

    def __init__(self, log_every_n_batches: int = 50) -> None:
        if log_every_n_batches <= 0:
            raise ValueError("log_every_n_batches must be a positive integer.")
        self.log_every_n_batches = log_every_n_batches
        self._batch_start_time: float | None = None
        self._last_batch_end_time: float | None = None

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        self._batch_start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del outputs
        if (batch_idx + 1) % self.log_every_n_batches != 0:
            return
        assert isinstance(batch, (list, tuple)), (
            "Expected batch to be a tuple/list of tensors."
        )
        batch_size = extract_batch_size(batch)
        batch_time = None
        if self._batch_start_time is not None:
            batch_time = time.perf_counter() - self._batch_start_time

        metrics: dict[str, float] = {}
        if batch_time is not None and batch_time > 0:
            metrics["batch_time_sec"] = batch_time
            metrics["throughput_samples_per_sec"] = float(batch_size) / batch_time
        if self._last_batch_end_time is not None:
            data_wait_time = self._batch_start_time - self._last_batch_end_time
            if data_wait_time >= 0:
                metrics["data_wait_time_sec"] = data_wait_time

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics["cpu_mem_rss_mb"] = float(rss_kb) / 1024.0

        if torch.cuda.is_available() and pl_module.device.type == "cuda":
            device = pl_module.device
            metrics["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(device) / 1e6
            metrics["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / 1e6

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
        self._last_batch_end_time = time.perf_counter()
