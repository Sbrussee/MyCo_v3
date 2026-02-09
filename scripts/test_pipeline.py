#!/usr/bin/env python3
"""
Lightweight pipeline smoke test for data -> model -> eval components.

This script uses in-memory fake slides and annotations to validate that:
  - The iterable dataset yields batches correctly.
  - The MoCo v3 model forward pass runs on sample data.
  - The evaluation embedding collection and probe training execute.
It also logs I/O throughput, CPU memory usage, and GPU memory usage (if available).
"""

from __future__ import annotations

import argparse
import logging
import random
import resource
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple
from unittest import mock
import torch
from PIL import Image
from tqdm import tqdm

from myco.data import SlideEntry, WSICellMoCoIterable
from myco.eval import EvalCallback, MosaicConfig, ProbeConfig, repr_metrics_np
from myco.model import MoCoV3Lit
from myco.utils import seed_all


@dataclass(frozen=True)
class FakeSlide:
    """Minimal OpenSlide-like object for testing."""

    size: int = 60

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> Image.Image:
        del location, level
        return Image.new("RGB", size, color=(127, 127, 127))

    def close(self) -> None:
        return None


def _fake_centroids(_: str) -> List[Tuple[float, float]]:
    return [(float(x), float(x)) for x in range(10)]


def _fake_open_slide(_: str) -> FakeSlide:
    return FakeSlide()


def _build_entries(num_slides: int) -> List[SlideEntry]:
    return [
        SlideEntry(
            slide_id=f"slide_{idx}",
            wsi_path=f"/fake/slide_{idx}.svs",
            ann_path=f"/fake/slide_{idx}.xml",
        )
        for idx in range(num_slides)
    ]


def _log_resource_usage(device: torch.device, prefix: str) -> None:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logging.info("%s CPU RSS: %.2f MB", prefix, rss_kb / 1024.0)
    if torch.cuda.is_available() and device.type == "cuda":
        logging.info(
            "%s GPU allocated: %.2f MB | reserved: %.2f MB",
            prefix,
            torch.cuda.memory_allocated(device) / 1e6,
            torch.cuda.memory_reserved(device) / 1e6,
        )


def _iter_batches(dataset: Iterable, num_batches: int) -> None:
    iterator = iter(dataset)
    for _ in range(num_batches):
        next(iterator)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test the MoCo pipeline end-to-end."
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--epoch_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    seed_all(args.seed)

    entries = _build_entries(num_slides=6)
    slide_labels = {entry.slide_id: int(idx % 2) for idx, entry in enumerate(entries)}

    with (
        mock.patch("myco.data.load_centroids", side_effect=_fake_centroids),
        mock.patch("myco.data.safe_open_slide", side_effect=_fake_open_slide),
    ):
        dataset = WSICellMoCoIterable(
            entries=entries, epoch_length=args.epoch_length, seed=args.seed
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=0
        )

        start = time.perf_counter()
        for _ in tqdm(range(args.batches), desc="Loading batches"):
            batch = next(iter(dataloader))
            assert isinstance(batch, (list, tuple)), (
                "Expected batch to be a tuple/list."
            )
            x1, x2 = batch
            assert x1.shape == x2.shape, "Augmented views must match shape."
        elapsed = time.perf_counter() - start
        logging.info(
            "Loaded %d batches in %.3f s (%.2f batches/s).",
            args.batches,
            elapsed,
            args.batches / elapsed,
        )

        model = MoCoV3Lit(
            init_ckpt="",
            lr=1e-4,
            weight_decay=0.0,
            temperature=0.2,
            proj_dim=32,
            mlp_hidden=64,
            base_m=0.99,
            epochs=1,
            steps_per_epoch=1,
            warmup_epochs=0,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        x1, _ = next(iter(dataloader))
        x1 = x1.to(device)
        embeddings = model.encode(x1).detach().cpu().numpy()
        metrics = repr_metrics_np(embeddings)
        logging.info("Representation metrics: %s", metrics)

        eval_cb = EvalCallback(
            entries=entries,
            slide_labels=slide_labels,
            output_dir=".",
            probe=ProbeConfig(
                cells_per_slide=5,
                probe_epochs=2,
                probe_lr=1e-3,
                slides_per_class=3,
                embed_batch_size=4,
                seed=args.seed,
            ),
            mosaic=MosaicConfig(
                method="tsne",
                max_points=50,
                point_size=4,
                thumb_size=4,
                random_state=args.seed,
            ),
        )
        with (
            mock.patch("myco.data.safe_open_slide", side_effect=_fake_open_slide),
            mock.patch("myco.data.load_centroids", side_effect=_fake_centroids),
        ):
            embeds_by_slide, labels, _, _ = eval_cb._collect_embeddings(
                model, device, rng=random.Random(args.seed)
            )
            probe_metrics = eval_cb._train_probe(embeds_by_slide, labels, device)
            logging.info("Probe metrics: %s", probe_metrics)

        _log_resource_usage(device, prefix="Pipeline test")


if __name__ == "__main__":
    main()
