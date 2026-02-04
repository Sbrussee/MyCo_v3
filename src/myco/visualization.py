"""Visualization utilities for embedding mosaics."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.manifold import TSNE


@dataclass(frozen=True)
class MosaicConfig:
    """Configuration for UMAP/TSNE patch mosaics."""

    method: str = "tsne"
    max_points: int = 400
    point_size: int = 10
    thumb_size: int = 12
    random_state: int = 0


def _build_umap_reducer(random_state: int):
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        raise RuntimeError(
            "UMAP requires numba modules compiled against NumPy <2. "
            "Please switch to --mosaic_method tsne or downgrade NumPy to <2."
        )
    if importlib.util.find_spec("umap") is None:
        raise RuntimeError(
            "UMAP is not installed. Install umap-learn or switch to --mosaic_method tsne."
        )
    umap_module = importlib.import_module("umap")
    reducer_cls = getattr(umap_module, "UMAP")
    return reducer_cls(n_components=2, random_state=random_state)


def _project_embeddings(
    embeddings: np.ndarray, method: str, random_state: int
) -> np.ndarray:
    """Project embeddings to 2D coordinates.

    Args:
        embeddings: Array with shape (num_points, embedding_dim).
        method: Projection method, either "umap" or "tsne".
        random_state: Random seed for deterministic projections.
    """
    assert isinstance(embeddings, np.ndarray), "Embeddings must be a NumPy array."
    assert embeddings.ndim == 2, (
        "Embeddings must have shape (num_points, embedding_dim)."
    )
    assert embeddings.shape[0] > 0, "Embeddings must contain at least one point."
    assert method in {"umap", "tsne"}, "Projection method must be 'umap' or 'tsne'."

    if method == "umap":
        reducer = _build_umap_reducer(random_state)
    else:
        reducer = TSNE(n_components=2, random_state=random_state, init="pca")
    return reducer.fit_transform(embeddings)


def create_patch_mosaic(
    embeddings: np.ndarray,
    patches: List[np.ndarray],
    config: MosaicConfig,
    title: str,
    output_path: str,
) -> None:
    """Create a patch mosaic plot using UMAP/TSNE embeddings.

    Args:
        embeddings: Array with shape (num_points, embedding_dim).
        patches: List of image patches with shape (H, W) or (H, W, C).
        config: Mosaic plotting configuration.
        title: Plot title.
        output_path: Output image path.
    """
    assert isinstance(embeddings, np.ndarray), "Embeddings must be a NumPy array."
    assert embeddings.ndim == 2, (
        "Embeddings must have shape (num_points, embedding_dim)."
    )
    assert isinstance(patches, list), "Patches must be a list of NumPy arrays."
    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings provided for mosaic plot.")

    n_points = min(config.max_points, embeddings.shape[0])
    embeddings = embeddings[:n_points]
    patches = patches[:n_points]
    assert len(patches) == n_points, (
        "Number of patches must match number of embeddings."
    )

    coords = _project_embeddings(embeddings, config.method, config.random_state)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=config.point_size, alpha=0.2, c="gray")

    for (x, y), patch in zip(coords, patches):
        assert isinstance(patch, np.ndarray), "Each patch must be a NumPy array."
        assert patch.ndim in {2, 3}, "Patch must have shape (H, W) or (H, W, C)."
        image = OffsetImage(patch, zoom=config.thumb_size / max(patch.shape[:2]))
        ab = AnnotationBbox(image, (x, y), frameon=False)
        ax.add_artist(ab)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
