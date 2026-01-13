"""Inference utilities for extracting embeddings from trained MoCo v3 weights."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torchvision.transforms import ToTensor

from .augment import RotationCrop40
from .data import SlideEntry, build_entries_from_dirs, load_centroids, safe_open_slide
from .model import MoCoV3Lit
from .utils import read_patch


def _load_weights(weights_path: str) -> Dict[str, torch.Tensor]:
    data = torch.load(weights_path, map_location="cpu")
    if "weights" in data:
        return data["weights"]
    return data


def _init_model(weights_path: str, device: torch.device) -> MoCoV3Lit:
    payload = _load_weights(weights_path)
    hparams = payload.get("hyperparameters", {})
    model = MoCoV3Lit(
        init_ckpt="",
        lr=hparams.get("lr", 2.5e-4),
        weight_decay=hparams.get("weight_decay", 0.05),
        temperature=hparams.get("temperature", 0.2),
        proj_dim=hparams.get("proj_dim", 256),
        mlp_hidden=hparams.get("mlp_hidden", 2048),
        base_m=hparams.get("m", 0.99),
        epochs=hparams.get("epochs", 1),
    )
    model.q_enc.load_state_dict(payload["q_enc"])
    model.q_proj.load_state_dict(payload["q_proj"])
    model.eval()
    model.to(device)
    return model


def _embed_slide(
    model: MoCoV3Lit,
    entry: SlideEntry,
    centroids: List[Tuple[float, float]],
    max_cells: int,
    device: torch.device,
) -> np.ndarray:
    rotcrop = RotationCrop40(big_size=60, out_size=40, degrees=360.0)
    totensor = ToTensor()
    rng = np.random.default_rng(0)
    embeddings: List[np.ndarray] = []
    slide = safe_open_slide(entry.wsi_path)
    try:
        for _ in range(max_cells):
            center = centroids[int(rng.integers(0, len(centroids)))]
            patch = read_patch(slide, center, 60)
            patch = rotcrop(patch)
            tensor = totensor(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                z = model.encode(tensor).squeeze(0).detach().cpu().numpy()
            embeddings.append(z)
    finally:
        slide.close()
    return np.stack(embeddings, axis=0)


def run_inference(
    wsi_dir: str,
    ann_dir: str,
    weights_path: str,
    out_dir: str,
    max_cells: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _init_model(weights_path, device)

    entries = build_entries_from_dirs(wsi_dir, ann_dir)
    for entry in entries:
        centroids = load_centroids(entry.ann_path)
        if not centroids:
            continue
        embeddings = _embed_slide(model, entry, centroids, max_cells, device)
        out_path = os.path.join(out_dir, f"{entry.slide_id}_embeddings.npy")
        np.save(out_path, embeddings)

    metadata = {
        "weights": weights_path,
        "entries": [entry.slide_id for entry in entries],
        "max_cells": max_cells,
    }
    with open(os.path.join(out_dir, "inference_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MoCo v3 embeddings from WSI slides.")
    parser.add_argument("--wsi_dir", required=True)
    parser.add_argument("--ann_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_cells", type=int, default=200)
    args = parser.parse_args()

    run_inference(
        wsi_dir=args.wsi_dir,
        ann_dir=args.ann_dir,
        weights_path=args.weights,
        out_dir=args.out_dir,
        max_cells=args.max_cells,
    )


if __name__ == "__main__":
    main()
