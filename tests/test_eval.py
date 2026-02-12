import math

import numpy as np
import torch

from myco.eval import EvalCallback, ProbeConfig
from myco.visualization import MosaicConfig


def _make_callback(seed: int = 0) -> EvalCallback:
    return EvalCallback(
        entries=[],
        slide_labels={},
        output_dir=".",
        probe=ProbeConfig(seed=seed, probe_epochs=1),
        mosaic=MosaicConfig(),
    )


def test_train_probe_handles_single_embedding_per_slide():
    callback = _make_callback(seed=0)
    embeds = {
        f"slide_{i}": np.full((1, 4), float(i), dtype=np.float32) for i in range(10)
    }
    labels = {f"slide_{i}": i % 2 for i in range(10)}

    metrics = callback._train_probe(embeds, labels, device=torch.device("cpu"))

    assert set(metrics) == {"probe_auc", "probe_bal_acc"}
    assert 0.0 <= metrics["probe_bal_acc"] <= 1.0


def test_train_probe_skips_empty_samples():
    callback = _make_callback(seed=1)
    embeds = {f"slide_{i}": np.empty((0, 4), dtype=np.float32) for i in range(6)}
    labels = {f"slide_{i}": i % 2 for i in range(6)}

    metrics = callback._train_probe(embeds, labels, device=torch.device("cpu"))

    assert math.isnan(metrics["probe_auc"])
    assert math.isnan(metrics["probe_bal_acc"])
