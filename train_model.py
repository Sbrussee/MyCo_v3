#!/usr/bin/env python3
"""
Training entrypoint for MoCo v3 on nuclei crops.

This script wires data, model, and evaluation callbacks together for
self-supervised training with PyTorch Lightning.
"""
from __future__ import annotations

import argparse
import os
import sys
import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from myco.data import CellDataModule, build_entries_from_dirs, read_slide_labels
from myco.eval import EvalCallback, MosaicConfig, ProbeConfig
from myco.model import MoCoV3Lit
from myco.utils import seed_all


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

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--probe_cells_per_slide", type=int, default=200)
    parser.add_argument("--probe_epochs", type=int, default=20)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_slides_per_class", type=int, default=150)

    parser.add_argument("--mosaic_method", choices=["umap", "tsne"], default="umap")
    parser.add_argument("--mosaic_max_points", type=int, default=400)
    parser.add_argument("--mosaic_point_size", type=int, default=10)
    parser.add_argument("--mosaic_thumb_size", type=int, default=12)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    seed_all(args.seed)

    entries = build_entries_from_dirs(args.wsi_dir, args.ann_dir)
    slide_labels = read_slide_labels(args.slide_labels)

    datamodule = CellDataModule(
        entries,
        epoch_length=args.epoch_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
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
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.outdir,
        filename="moco-{epoch:03d}",
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
    )

    probe_cfg = ProbeConfig(
        cells_per_slide=args.probe_cells_per_slide,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        slides_per_class=args.probe_slides_per_class,
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
        strategy=DDPStrategy(find_unused_parameters=False) if args.devices > 1 else "auto",
        precision=args.precision,
        accumulate_grad_batches=args.accum,
        callbacks=[checkpoint_cb, eval_cb],
        log_every_n_steps=50,
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
