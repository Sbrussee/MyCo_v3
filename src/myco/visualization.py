"""Visualization utilities for embedding mosaics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.manifold import TSNE
from umap import UMAP


@dataclass(frozen=True)
class MosaicConfig:
    """Configuration for UMAP/TSNE patch mosaics."""

    method: str = "umap"
    max_points: int = 400
    point_size: int = 10
    thumb_size: int = 12
    random_state: int = 0


def _project_embeddings(embeddings: np.ndarray, method: str, random_state: int) -> np.ndarray:
    if method == "umap":
        reducer = UMAP(n_components=2, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state, init="pca")
    else:
        raise ValueError(f"Unknown projection method: {method}")
    return reducer.fit_transform(embeddings)


def create_patch_mosaic(
    embeddings: np.ndarray,
    patches: List[np.ndarray],
    config: MosaicConfig,
    title: str,
    output_path: str,
) -> None:
    """Create a patch mosaic plot using UMAP/TSNE embeddings."""
    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings provided for mosaic plot.")

    n_points = min(config.max_points, embeddings.shape[0])
    embeddings = embeddings[:n_points]
    patches = patches[:n_points]

    coords = _project_embeddings(embeddings, config.method, config.random_state)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=config.point_size, alpha=0.2, c="gray")

    for (x, y), patch in zip(coords, patches):
        image = OffsetImage(patch, zoom=config.thumb_size / max(patch.shape[:2]))
        ab = AnnotationBbox(image, (x, y), frameon=False)
        ax.add_artist(ab)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
