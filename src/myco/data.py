"""Dataset and IO utilities for WSI-centric nucleus cropping."""
from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import LightningDataModule as PLDataModule
    from torch.utils.data import DataLoader, IterableDataset
else:
    class IterableDataset:  # type: ignore[no-redef]
        """Fallback IterableDataset base when torch is unavailable."""

        pass

    class PLDataModule:  # type: ignore[no-redef]
        """Fallback LightningDataModule base when Lightning is unavailable."""

        pass



@dataclass(frozen=True)
class SlideEntry:
    """Reference to a WSI and its annotation file."""

    slide_id: str
    wsi_path: str
    ann_path: str


def read_slide_labels(path: str) -> Dict[str, int]:
    """Read slide labels from CSV/JSON into a slide_id -> {0,1} mapping."""
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        output: Dict[str, int] = {}
        for slide_id, label in data.items():
            if isinstance(label, str):
                output[slide_id] = 1 if label.strip().upper() == "MF" else 0
            else:
                output[slide_id] = int(label)
        return output

    output: Dict[str, int] = {}
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            slide_id = row["slide_id"]
            label = row["label"].strip().upper()
            if label in ["MF", "1", "TRUE", "T"]:
                output[slide_id] = 1
            elif label in ["BID", "0", "FALSE", "F"]:
                output[slide_id] = 0
            else:
                output[slide_id] = int(float(label))
    return output


def parse_geojson_centroids(path: str) -> List[Tuple[float, float]]:
    """Parse centroids from a GeoJSON or JSON annotation file."""
    from shapely.geometry import shape

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    coords: List[Tuple[float, float]] = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if geom is None:
            continue
        geom_obj = shape(geom)
        centroid = geom_obj if geom_obj.geom_type == "Point" else geom_obj.centroid
        coords.append((float(centroid.x), float(centroid.y)))
    return coords


def parse_xml_centroids(path: str) -> List[Tuple[float, float]]:
    """Parse centroids from XML annotations (ASAP/QuPath style)."""
    from lxml import etree

    tree = etree.parse(path)
    root = tree.getroot()
    coords: List[Tuple[float, float]] = []
    for coord in root.findall(".//Coordinate"):
        x_val = coord.get("X") or coord.get("x")
        y_val = coord.get("Y") or coord.get("y")
        if x_val is None or y_val is None:
            continue
        coords.append((float(x_val), float(y_val)))
    return coords


def load_centroids(path: str) -> List[Tuple[float, float]]:
    """Load centroids from XML or GeoJSON annotations."""
    lower = path.lower()
    if lower.endswith(".geojson") or lower.endswith(".json"):
        return parse_geojson_centroids(path)
    if lower.endswith(".xml"):
        return parse_xml_centroids(path)
    raise ValueError(f"Unsupported annotation format: {path}")


def safe_open_slide(wsi_path: str):
    """Open a WSI path with OpenSlide, raising a clear error if missing."""
    import openslide

    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI path not found: {wsi_path}")
    return openslide.OpenSlide(wsi_path)


def build_entries_from_dirs(
    wsi_dir: str,
    ann_dir: str,
    wsi_exts: Iterable[str] = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs"),
) -> List[SlideEntry]:
    """Match WSI files with annotation files by shared stem name."""
    wsi_dir_path = Path(wsi_dir)
    ann_dir_path = Path(ann_dir)

    wsis: List[Path] = []
    for ext in wsi_exts:
        wsis.extend(wsi_dir_path.glob(f"*{ext}"))
    if not wsis:
        raise RuntimeError(f"No WSIs found in {wsi_dir} with extensions {wsi_exts}")

    wsi_map = {path.stem: path for path in wsis}
    annotations = list(ann_dir_path.glob("*.xml"))
    annotations += list(ann_dir_path.glob("*.geojson"))
    annotations += list(ann_dir_path.glob("*.json"))
    ann_map = {path.stem: path for path in annotations}

    entries: List[SlideEntry] = []
    for stem, wsi_path in wsi_map.items():
        ann_path = ann_map.get(stem)
        if ann_path is None:
            continue
        entries.append(SlideEntry(slide_id=stem, wsi_path=str(wsi_path), ann_path=str(ann_path)))

    if not entries:
        raise RuntimeError("No matched WSI/annotation pairs by stem name.")
    return entries


class WSICellMoCoIterable(IterableDataset):
    """Iterable dataset yielding two augmented 40x40 nucleus crops."""

    def __init__(
        self,
        entries: List[SlideEntry],
        epoch_length: int,
        seed: int,
        out_size: int = 40,
        big_size: int = 60,
    ) -> None:
        super().__init__()
        self.all_entries = entries
        self.epoch_length = epoch_length
        self.seed = seed
        self.out_size = out_size
        self.big_size = big_size

        from .augment import RotationCrop40, build_lemon_a1_gray_transform

        self.rotcrop = RotationCrop40(big_size=big_size, out_size=out_size, degrees=360.0)
        self.aug = build_lemon_a1_gray_transform(img_size=out_size)

        self.centroids: Dict[str, List[Tuple[float, float]]] = {}
        for entry in entries:
            self.centroids[entry.slide_id] = load_centroids(entry.ann_path)

    def __iter__(self):
        import torch
        from .utils import read_patch

        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        entries = self.all_entries[rank::world_size]
        if not entries:
            entries = self.all_entries

        rng = random.Random(self.seed + 1000 * rank + 10 * worker_id)

        n_yield = self.epoch_length // (world_size * num_workers)
        for _ in range(n_yield):
            entry = rng.choice(entries)
            cents = self.centroids.get(entry.slide_id, [])
            if not cents:
                continue
            center = rng.choice(cents)
            slide = safe_open_slide(entry.wsi_path)
            try:
                patch = read_patch(slide, center, self.big_size)
            finally:
                slide.close()

            img40 = self.rotcrop(patch)
            view1 = self.aug(img40)
            view2 = self.aug(img40)
            yield view1, view2


class CellDataModule(PLDataModule):
    """Lightning DataModule for nucleus crop sampling."""

    def __init__(
        self, entries: List[SlideEntry], epoch_length: int, batch_size: int, num_workers: int, seed: int
    ) -> None:
        super().__init__()
        self.entries = entries
        self.epoch_length = epoch_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def train_dataloader(self) -> DataLoader:
        from torch.utils.data import DataLoader

        dataset = WSICellMoCoIterable(self.entries, self.epoch_length, seed=self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
