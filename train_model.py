#!/usr/bin/env python3
"""
Training entrypoint for MoCo v3 on nuclei crops.

This script wires data, model, and evaluation callbacks together for
self-supervised training with PyTorch Lightning.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from myco.callbacks import BatchMetricsLogger  # noqa: E402
from myco.data import (  # noqa: E402
    CellDataModule,
    DebugSampleConfig,
    build_entries_from_dirs,
    read_slide_labels,
)
from myco.eval import EvalCallback, MosaicConfig, ProbeConfig  # noqa: E402
from myco.model import MoCoV3Lit  # noqa: E402
from myco.utils import seed_all  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MoCo v3 on nuclei crops.")
    parser.add_argument("--wsi_dir", required=True)
    parser.add_argument("--ann_dir", required=True)
    parser.add_argument("--slide_labels", required=True)

    parser.add_argument("--outdir", required=True)
    parser.add_argument("--init_ckpt", default="")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--epoch_length", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--accum", type=int, default=16, help="accumulate_grad_batches")

    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--m", type=float, default=0.99)
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Linear warmup epochs for the MoCo v3 cosine learning-rate schedule.",
    )

    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--mlp_hidden", type=int, default=2048)
    parser.add_argument("--img_size", type=int, default=40)
    parser.add_argument("--big_size", type=int, default=60)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--centroid_cache_dir",
        default="",
        help="Optional disk directory for centroid cache arrays (defaults to system tmp).",
    )
    parser.add_argument(
        "--in_memory_centroid_limit",
        type=int,
        default=8,
        help="Maximum number of slide centroid arrays kept mmap-open in memory.",
    )
    parser.add_argument(
        "--log_every_n_batches",
        type=int,
        default=50,
        help="Log throughput/memory metrics every N batches.",
    )
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--debug_dir",
        default="",
        help="Optional directory for saving debug samples (defaults to outdir/debug_samples).",
    )
    parser.add_argument(
        "--debug_samples",
        type=int,
        default=5,
        help="Number of debug samples to save from the data pipeline.",
    )

    parser.add_argument("--probe_cells_per_slide", type=int, default=200)
    parser.add_argument("--probe_epochs", type=int, default=20)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_slides_per_class", type=int, default=150)
    parser.add_argument(
        "--probe_embed_batch_size",
        type=int,
        default=256,
        help="Batch size for slide-level embedding extraction during evaluation.",
    )

    parser.add_argument(
        "--mosaic_method",
        choices=["umap", "tsne"],
        default="tsne",
        help="Embedding projection method for mosaics (UMAP requires NumPy <2).",
    )
    parser.add_argument("--mosaic_max_points", type=int, default=400)
    parser.add_argument("--mosaic_point_size", type=int, default=10)
    parser.add_argument("--mosaic_thumb_size", type=int, default=12)
    return parser


def build_logger(outdir: str) -> CSVLogger:
    """Create a CSV logger for metrics and performance logging."""
    return CSVLogger(save_dir=outdir, name="logs")


def build_callbacks(
    outdir: str,
    eval_cb: pl.Callback,
    log_every_n_batches: int,
) -> list[pl.Callback]:
    """Create training callbacks for checkpointing, metrics, and device stats."""
    checkpoint_cb = ModelCheckpoint(
        dirpath=outdir,
        filename="moco-{epoch:03d}",
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    device_monitor = DeviceStatsMonitor()
    progress_bar = TQDMProgressBar(refresh_rate=1)
    batch_logger = BatchMetricsLogger(log_every_n_batches=log_every_n_batches)
    return [
        checkpoint_cb,
        eval_cb,
        lr_monitor,
        device_monitor,
        progress_bar,
        batch_logger,
    ]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_argparser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    seed_all(args.seed)
    assert args.img_size > 0, "img_size must be positive."
    assert args.big_size >= args.img_size, "big_size must be >= img_size."

    logger = logging.getLogger(__name__)
    logger.info("Training configuration: %s", vars(args))

    entries = build_entries_from_dirs(args.wsi_dir, args.ann_dir)
    slide_labels = read_slide_labels(args.slide_labels)
    logger.info("Loaded %d slide labels from %s.", len(slide_labels), args.slide_labels)
    debug_dir = args.debug_dir or os.path.join(args.outdir, "debug_samples")
    debug_config = None
    if args.debug_samples > 0:
        debug_config = DebugSampleConfig(
            output_dir=Path(debug_dir), max_samples=args.debug_samples
        )

    datamodule = CellDataModule(
        entries,
        epoch_length=args.epoch_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        out_size=args.img_size,
        big_size=args.big_size,
        debug_config=debug_config,
        centroid_cache_dir=args.centroid_cache_dir or None,
        in_memory_centroid_limit=args.in_memory_centroid_limit,
    )

    world_size = max(1, args.devices)
    steps_per_epoch = math.ceil(args.epoch_length / (args.batch_size * world_size))

    model = MoCoV3Lit(
        init_ckpt=args.init_ckpt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        proj_dim=args.proj_dim,
        mlp_hidden=args.mlp_hidden,
        base_m=args.m,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        img_size=args.img_size,
        big_size=args.big_size,
    )

    trainer_logger = build_logger(args.outdir)

    probe_cfg = ProbeConfig(
        cells_per_slide=args.probe_cells_per_slide,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        slides_per_class=args.probe_slides_per_class,
        embed_batch_size=args.probe_embed_batch_size,
        seed=args.seed,
    )
    mosaic_cfg = MosaicConfig(
        method=args.mosaic_method,
        max_points=args.mosaic_max_points,
        point_size=args.mosaic_point_size,
        thumb_size=args.mosaic_thumb_size,
        random_state=args.seed,
    )
    eval_cb = EvalCallback(
        entries=entries,
        slide_labels=slide_labels,
        output_dir=args.outdir,
        probe=probe_cfg,
        mosaic=mosaic_cfg,
    )

    trainer = pl.Trainer(
        default_root_dir=args.outdir,
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False)
        if args.devices > 1
        else "auto",
        precision=args.precision,
        accumulate_grad_batches=args.accum,
        callbacks=build_callbacks(args.outdir, eval_cb, args.log_every_n_batches),
        log_every_n_steps=args.log_every_n_batches,
        enable_checkpointing=True,
        logger=trainer_logger,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
