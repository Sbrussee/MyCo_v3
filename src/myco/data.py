"""Dataset and IO utilities for WSI-centric nucleus cropping."""
from __future__ import annotations

import csv
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pytorch_lightning import LightningDataModule as PLDataModule
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class SlideEntry:
    """Reference to a WSI and its annotation file."""

    slide_id: str
    wsi_path: str
    ann_path: str


def read_slide_labels(path: str) -> Dict[str, int]:
    """Read slide labels from CSV/JSON into a slide_id -> {0,1} mapping.

    Expected columns/keys:
      - slide_id or slide
      - label or category

    Only keeps slides explicitly labeled as 'MF' or 'BID' (case-insensitive).
    All other labels are ignored.

    Supports comma- or semicolon-separated CSV (auto-detected).
    """
    def _map_label(raw: object) -> Optional[int]:
        if raw is None:
            return None
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return int(raw)
        text = str(raw).strip().upper()
        if text in {"0", "1"}:
            return int(text)
        if text == "MF":
            return 1
        if text == "BID":
            return 0
        return None  # ignore everything else

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        output: Dict[str, int] = {}
        # Support either {slide_id: label} or [{"slide_id": ..., "label": ...}, ...]
        if isinstance(data, dict):
            for slide_id, label_raw in data.items():
                mapped = _map_label(label_raw)
                if mapped is None:
                    continue
                sid = str(slide_id).strip()
                if not sid:
                    continue
                output[sid] = mapped
            return output

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                sid_raw = item.get("slide_id") or item.get("slide")
                label_raw = item.get("label") or item.get("category")
                mapped = _map_label(label_raw)
                if mapped is None or sid_raw is None:
                    continue
                sid = str(sid_raw).strip()
                if not sid:
                    continue
                output[sid] = mapped
            return output

        raise ValueError("Unsupported JSON structure for slide labels.")

    output: Dict[str, int] = {}
    with open(path, "r", newline="", encoding="utf-8") as handle:
        sample = handle.read(8192)
        handle.seek(0)

        delimiter = ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            delimiter = dialect.delimiter
        except csv.Error:
            if ";" in sample and "," not in sample:
                delimiter = ";"

        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        assert fieldnames, "CSV must include headers."

        field_map = {name.lower(): name for name in fieldnames}
        slide_key = field_map.get("slide_id") or field_map.get("slide")
        assert slide_key is not None, "CSV must contain a 'slide_id' or 'slide' column."
        label_key = field_map.get("label") or field_map.get("category")
        assert label_key is not None, "CSV must contain a 'label' or 'category' column."

        for row in reader:
            slide_id_raw = row.get(slide_key)
            label_raw = row.get(label_key)

            mapped = _map_label(label_raw)
            if mapped is None or slide_id_raw is None:
                continue

            slide_id = slide_id_raw.strip()
            if not slide_id:
                continue

            output[slide_id] = mapped

    return output


def parse_geojson_centroids_from_payload(data: Dict[str, object]) -> List[Tuple[float, float]]:
    """Parse centroids from a GeoJSON payload.

    Expected payload format:
      - {"features": [{"geometry": {"type": "Point"|"Polygon"|...}}]}
    """
    from shapely.geometry import shape

    assert isinstance(data, dict), "GeoJSON payload must be a dictionary."
    coords: List[Tuple[float, float]] = []
    for feat in data.get("features", []):
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry")
        if geom is None:
            continue
        geom_obj = shape(geom)
        centroid = geom_obj if geom_obj.geom_type == "Point" else geom_obj.centroid
        coords.append((float(centroid.x), float(centroid.y)))
    return coords


def parse_geojson_centroids(path: str) -> List[Tuple[float, float]]:
    """Parse centroids from a GeoJSON or JSON annotation file."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return parse_geojson_centroids_from_payload(data)


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


def load_centroids(path: str, slide_path: Optional[str] = None) -> List[Tuple[float, float]]:
    """Load centroids from XML, GeoJSON, or HistoPLUS JSON annotations.

    Parameters
    ----------
    path : str
        Path to the annotation file.
    slide_path : str, optional
        WSI path required for HistoPLUS JSON annotations (tile-local coordinates).
    """
    cache_path = _centroid_cache_path(path)
    if cache_path.exists():
        return _read_centroid_cache(cache_path)

    lower = path.lower()
    if lower.endswith(".geojson"):
        centroids = parse_geojson_centroids(path)
    elif lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "cell_masks" in data:
            if slide_path is None:
                raise ValueError("slide_path must be provided for HistoPLUS JSON annotations.")
            from .histoplus import histoplus_centroids_from_payload

            cell_masks = data.get("cell_masks", [])
            centroids = histoplus_centroids_from_payload(
                cell_masks=cell_masks,
                slide_path=slide_path,
                apply_bounds_offset=False,
                progress=False,
            )
        else:
            centroids = parse_geojson_centroids_from_payload(data)
    elif lower.endswith(".xml"):
        centroids = parse_xml_centroids(path)
    else:
        raise ValueError(f"Unsupported annotation format: {path}")

    _write_centroid_cache(cache_path, centroids)
    return centroids


def _centroid_cache_path(path: str) -> Path:
    """Return the cache path used to store serialized centroids for an annotation."""
    ann_path = Path(path)
    return ann_path.with_suffix(ann_path.suffix + ".centroids.json")


def _read_centroid_cache(path: Path) -> List[Tuple[float, float]]:
    """Read cached centroid coordinates from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        cached = json.load(handle)
    return [(float(item[0]), float(item[1])) for item in cached]


def _write_centroid_cache(path: Path, centroids: List[Tuple[float, float]]) -> None:
    """Persist centroid coordinates to disk for future reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [[float(x), float(y)] for x, y in centroids]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


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
            self.centroids[entry.slide_id] = load_centroids(entry.ann_path, slide_path=entry.wsi_path)
        self.valid_entries = [entry for entry in entries if self.centroids.get(entry.slide_id)]
        total_centroids = sum(len(self.centroids.get(entry.slide_id, [])) for entry in entries)
        logger.info(
            "WSI dataset initialized with %d entries (%d with centroids, %d total centroids).",
            len(entries),
            len(self.valid_entries),
            total_centroids,
        )
        if not self.valid_entries:
            raise ValueError(
                "No valid entries with centroids found. Check annotation files and formats."
            )

    def __iter__(self):
        import torch
        from .utils import read_patch

        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        entries = self.valid_entries[rank::world_size]
        if not entries:
            entries = self.valid_entries

        rng = random.Random(self.seed + 1000 * rank + 10 * worker_id)

        n_yield = self.epoch_length // (world_size * num_workers)
        if n_yield <= 0:
            logger.warning(
                "Epoch length %d too small for world size %d and %d workers; no samples will be yielded.",
                self.epoch_length,
                world_size,
                num_workers,
            )
            return
        for _ in range(n_yield):
            entry = rng.choice(entries)
            centroids = self.centroids.get(entry.slide_id, [])
            assert centroids, f"Expected non-empty centroids for slide {entry.slide_id}."
            center = rng.choice(centroids)
            slide = safe_open_slide(entry.wsi_path)
            try:
                patch = read_patch(slide, center, self.big_size)
            finally:
                slide.close()

            img40 = self.rotcrop(patch)
            view1 = self.aug(img40)
            view2 = self.aug(img40)
            assert isinstance(view1, torch.Tensor), "Augmentation pipeline must return torch.Tensor."
            assert isinstance(view2, torch.Tensor), "Augmentation pipeline must return torch.Tensor."
            assert view1.shape == view2.shape, "Paired views must have identical shapes."
            assert view1.ndim == 3, f"Expected CHW tensor, got shape {tuple(view1.shape)}."
            expected_hw = (self.out_size, self.out_size)
            assert view1.shape[-2:] == expected_hw, (
                f"Expected HxW {expected_hw}, got {tuple(view1.shape[-2:])}."
            )
            yield view1, view2


class CellDataModule(PLDataModule):
    """Lightning DataModule for nucleus crop sampling."""

    def __init__(
        self, entries: List[SlideEntry], epoch_length: int, batch_size: int, num_workers: int, seed: int
    ) -> None:
        super().__init__()
        assert isinstance(entries, list), "entries must be a list of SlideEntry objects."
        assert all(isinstance(entry, SlideEntry) for entry in entries), "entries must contain SlideEntry objects."
        assert epoch_length > 0, "epoch_length must be a positive integer."
        assert batch_size > 0, "batch_size must be a positive integer."
        assert num_workers >= 0, "num_workers must be non-negative."
        self.entries = entries
        self.epoch_length = epoch_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def train_dataloader(self) -> DataLoader:
        dataset = WSICellMoCoIterable(self.entries, self.epoch_length, seed=self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
