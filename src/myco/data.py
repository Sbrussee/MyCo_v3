"""Dataset and IO utilities for WSI-centric nucleus cropping."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import tempfile
from dataclasses import dataclass
from collections import OrderedDict
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pytorch_lightning import LightningDataModule as PLDataModule
from torch.utils.data import DataLoader, IterableDataset

from .annotations import (
    filter_centroids_to_bounds,
    get_slide_geometry,
    load_centroids_from_json,
    parse_geojson_centroids_from_payload,
    parse_xml_centroids,
    read_annotation_text,
    summarize_annotation_payload,
    truncate_payload,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlideEntry:
    """Reference to a WSI and its annotation file."""

    slide_id: str
    wsi_path: str
    ann_path: str


@dataclass(frozen=True)
class DebugSampleConfig:
    """Configuration for saving debug samples from the data pipeline."""

    output_dir: Path
    max_samples: int = 0
    save_augmentation_examples: bool = True
    augmentation_seed: int = 0
    augmentation_dirname: str = "augmentations"


def read_slide_labels(
    path: str, allowed_datasets: Optional[Sequence[str]] = None
) -> Dict[str, int]:
    """Read slide labels from CSV/JSON into a slide_id -> {0,1} mapping.

    Expected columns/keys:
      - slide_id or slide
      - label or category

    Only keeps slides explicitly labeled as 'MF' or 'BID' (case-insensitive).
    All other labels are ignored.

    Supports comma- or semicolon-separated CSV (auto-detected).

    Parameters
    ----------
    path : str
        Path to labels CSV or JSON.
    allowed_datasets : sequence[str], optional
        If provided, only rows with ``dataset`` in this allow-list are kept
        (case-insensitive).
    """

    allowed_dataset_set = None
    if allowed_datasets is not None:
        allowed_dataset_set = {str(value).strip().upper() for value in allowed_datasets}

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
        if isinstance(data, dict):
            sid_raw = data.get("slide_id") or data.get("slide")
            label_raw = data.get("label") or data.get("category")
            mapped = _map_label(label_raw)
            if sid_raw is not None and mapped is not None:
                sid = str(sid_raw).strip()
                if sid:
                    output[sid] = mapped
                    return output
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
                if allowed_dataset_set is not None:
                    dataset_raw = item.get("dataset")
                    if (
                        dataset_raw is None
                        or str(dataset_raw).strip().upper() not in allowed_dataset_set
                    ):
                        mapped = None
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
        dataset_key = field_map.get("dataset")
        if allowed_dataset_set is not None:
            assert dataset_key is not None, (
                "CSV must contain a 'dataset' column when allowed_datasets is set."
            )

        for row in reader:
            slide_id_raw = row.get(slide_key)
            label_raw = row.get(label_key)

            if allowed_dataset_set is not None and dataset_key is not None:
                dataset_raw = row.get(dataset_key)
                if (
                    dataset_raw is None
                    or str(dataset_raw).strip().upper() not in allowed_dataset_set
                ):
                    continue

            mapped = _map_label(label_raw)
            if mapped is None or slide_id_raw is None:
                continue

            slide_id = slide_id_raw.strip()
            if not slide_id:
                continue

            output[slide_id] = mapped

    return output


def _summarize_centroids(centroids: List[Tuple[float, float]]) -> str:
    """Summarize centroid coordinates for logging."""
    if not centroids:
        return "count=0"
    xs = [coord[0] for coord in centroids]
    ys = [coord[1] for coord in centroids]
    return (
        f"count={len(centroids)} x_range=({min(xs):.2f},{max(xs):.2f}) "
        f"y_range=({min(ys):.2f},{max(ys):.2f})"
    )


def load_centroids(
    path: str, slide_path: Optional[str] = None
) -> List[Tuple[float, float]]:
    """Load centroids from XML, GeoJSON, or HistoPLUS JSON annotations.

    Parameters
    ----------
    path : str
        Path to the annotation file.
    slide_path : str, optional
        WSI path required for HistoPLUS JSON annotations (tile-local coordinates).
    """

    lower = path.lower()
    raw_text: Optional[str] = None
    if lower.endswith(".geojson"):
        raw_text = read_annotation_text(path)
        data = json.loads(raw_text)
        centroids = parse_geojson_centroids_from_payload(data)
    elif lower.endswith(".json"):
        raw_text = read_annotation_text(path)
        data = json.loads(raw_text)
        centroids = load_centroids_from_json(
            path=path,
            data=data,
            slide_path=slide_path,
            progress=False,
        )
    elif lower.endswith(".xml"):
        centroids = parse_xml_centroids(path)
    else:
        raise ValueError(f"Unsupported annotation format: {path}")

    if not centroids:
        summary = ""
        if lower.endswith((".json", ".geojson")):
            summary = f" Payload summary: {summarize_annotation_payload(data)}."
        if raw_text is None and lower.endswith(".xml"):
            raw_text = read_annotation_text(path)
        payload_text = ""
        if raw_text is not None:
            payload_text = f"\nAnnotation payload:\n{truncate_payload(raw_text)}"
        logger.warning(
            "No centroids parsed from %s (format=%s).%s%s",
            path,
            Path(path).suffix,
            summary,
            payload_text,
        )

    if slide_path is not None:
        try:
            geometry = get_slide_geometry(slide_path)
            centroids = filter_centroids_to_bounds(centroids, geometry)
        except Exception as exc:  # noqa: BLE001 - OpenSlide failure should not stop parsing.
            logger.warning("Failed to validate centroids against slide bounds: %s", exc)

    logger.info(
        "Parsed centroids from %s (format=%s, %s).",
        path,
        Path(path).suffix,
        _summarize_centroids(centroids),
    )
    return centroids


def safe_open_slide(wsi_path: str):
    """Open a WSI path with OpenSlide, raising a clear error if missing."""
    import openslide

    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI path not found: {wsi_path}")
    return openslide.OpenSlide(wsi_path)


def _save_debug_sample(
    config: DebugSampleConfig,
    sample_idx: int,
    entry: SlideEntry,
    center: Tuple[float, float],
    patch,
    base_crop,
    view1,
    view2,
    out_size: int,
    big_size: int,
) -> None:
    """Save debug images + metadata for a sampled patch."""
    import torch
    from torchvision.transforms.functional import to_pil_image

    assert config.max_samples >= 0, "max_samples must be non-negative."
    assert patch.size == (big_size, big_size), (
        f"Expected patch size {(big_size, big_size)}."
    )
    assert isinstance(view1, torch.Tensor), "view1 must be a torch.Tensor."
    assert isinstance(view2, torch.Tensor), "view2 must be a torch.Tensor."
    assert view1.shape == view2.shape, "Debug views must match shapes."
    assert view1.shape[-2:] == (out_size, out_size), "Unexpected debug view size."
    assert base_crop.size == (out_size, out_size), (
        f"Expected base crop size {(out_size, out_size)}, got {base_crop.size}."
    )

    safe_slide_id = entry.slide_id.replace(os.sep, "_")
    sample_prefix = f"{safe_slide_id}_sample_{sample_idx:04d}"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    patch_path = config.output_dir / f"{sample_prefix}_patch.png"
    base_crop_path = config.output_dir / f"{sample_prefix}_base_crop.png"
    view1_path = config.output_dir / f"{sample_prefix}_view1.png"
    view2_path = config.output_dir / f"{sample_prefix}_view2.png"
    metadata_path = config.output_dir / f"{sample_prefix}_meta.json"

    patch.save(patch_path)
    base_crop.save(base_crop_path)
    to_pil_image(view1.detach().cpu().clamp(0, 1)).save(view1_path)
    to_pil_image(view2.detach().cpu().clamp(0, 1)).save(view2_path)

    aug_dir = None
    aug_seed = None
    if config.save_augmentation_examples:
        from .augment import save_augmentation_examples

        aug_seed = config.augmentation_seed + sample_idx
        aug_dir = config.output_dir / f"{sample_prefix}_{config.augmentation_dirname}"
        save_augmentation_examples(
            base_crop,
            output_dir=aug_dir,
            img_size=out_size,
            seed=aug_seed,
        )

    metadata = {
        "slide_id": entry.slide_id,
        "wsi_path": entry.wsi_path,
        "ann_path": entry.ann_path,
        "center": [float(center[0]), float(center[1])],
        "patch_size": [int(big_size), int(big_size)],
        "view_shape": list(view1.shape),
        "patch_path": str(patch_path),
        "base_crop_path": str(base_crop_path),
        "view1_path": str(view1_path),
        "view2_path": str(view2_path),
        "augmentation_examples_dir": str(aug_dir) if aug_dir is not None else None,
        "augmentation_seed": aug_seed,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info(
        "Saved debug sample %d for slide %s to %s.",
        sample_idx,
        entry.slide_id,
        config.output_dir,
    )


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
    logger.info("Found %d WSI files in %s.", len(wsis), wsi_dir)

    wsi_map = {path.stem: path for path in wsis}
    annotations = list(ann_dir_path.glob("*.xml"))
    annotations += list(ann_dir_path.glob("*.geojson"))
    annotations += list(ann_dir_path.glob("*.json"))
    ann_map = {path.stem: path for path in annotations}
    logger.info("Found %d annotation files in %s.", len(annotations), ann_dir)

    entries: List[SlideEntry] = []
    for stem, wsi_path in wsi_map.items():
        ann_path = ann_map.get(stem)
        if ann_path is None:
            continue
        entries.append(
            SlideEntry(slide_id=stem, wsi_path=str(wsi_path), ann_path=str(ann_path))
        )

    unmatched_wsi = sorted(set(wsi_map.keys()) - set(ann_map.keys()))
    unmatched_ann = sorted(set(ann_map.keys()) - set(wsi_map.keys()))
    for stem in unmatched_wsi:
        logger.info("No annotation found for WSI stem %s.", stem)
    for stem in unmatched_ann:
        logger.info("No WSI found for annotation stem %s.", stem)

    if not entries:
        raise RuntimeError("No matched WSI/annotation pairs by stem name.")
    logger.info(
        "Matched %d WSI/annotation pairs (wsi_dir=%s ann_dir=%s).",
        len(entries),
        wsi_dir,
        ann_dir,
    )
    for entry in entries:
        logger.info(
            "Entry matched: slide_id=%s wsi_path=%s ann_path=%s",
            entry.slide_id,
            entry.wsi_path,
            entry.ann_path,
        )
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
        debug_config: Optional[DebugSampleConfig] = None,
        centroid_cache_dir: Optional[str] = None,
        in_memory_centroid_limit: int = 8,
    ) -> None:
        super().__init__()
        assert out_size > 0, "out_size must be positive."
        assert big_size > 0, "big_size must be positive."
        assert big_size >= out_size, "big_size must be >= out_size."
        self.all_entries = entries
        self.epoch_length = epoch_length
        self.seed = seed
        self.out_size = out_size
        self.big_size = big_size
        self.debug_config = debug_config
        self._debug_count = 0
        self.centroid_cache_dir = Path(
            centroid_cache_dir
            if centroid_cache_dir
            else os.path.join(tempfile.gettempdir(), "myco_v3_centroid_cache")
        )
        assert in_memory_centroid_limit > 0, (
            "in_memory_centroid_limit must be positive."
        )
        self.in_memory_centroid_limit = in_memory_centroid_limit
        self._centroid_mem_cache: OrderedDict[str, np.memmap] = OrderedDict()
        self._slide_cache: OrderedDict[str, object] = OrderedDict()
        self._slide_cache_limit = 2

        from .augment import RotationCrop40, build_lemon_a1_gray_transform

        self.rotcrop = RotationCrop40(
            big_size=big_size, out_size=out_size, degrees=360.0
        )
        self.aug = build_lemon_a1_gray_transform(img_size=out_size)

        self.centroids: Dict[str, List[Tuple[float, float]]] = {}
        self._centroid_count_by_slide: Dict[str, int] = {}
        self._centroid_cache_path_by_slide: Dict[str, str] = {}
        self.centroid_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            from tqdm.auto import tqdm

            entry_iter = tqdm(entries, desc="Loading centroids", unit="slide")
        except Exception:  # noqa: BLE001 - tqdm is optional at runtime.
            entry_iter = entries
        for entry in entry_iter:
            centroids = load_centroids(entry.ann_path, slide_path=entry.wsi_path)
            self.centroids[entry.slide_id] = []
            self._centroid_count_by_slide[entry.slide_id] = len(centroids)
            if centroids:
                cache_path = self._build_centroid_cache_path(entry)
                centroid_array = np.asarray(centroids, dtype=np.float32)
                assert centroid_array.ndim == 2 and centroid_array.shape[1] == 2, (
                    "Centroid cache arrays must have shape [N, 2]."
                )
                np.save(cache_path, centroid_array)
                self._centroid_cache_path_by_slide[entry.slide_id] = cache_path
                del centroid_array
            del centroids
        self.valid_entries = [
            entry
            for entry in entries
            if self._centroid_count_by_slide.get(entry.slide_id, 0) > 0
        ]
        self._cum_centroid_counts: List[int] = []
        running_total = 0
        for entry in self.valid_entries:
            running_total += self._centroid_count_by_slide[entry.slide_id]
            self._cum_centroid_counts.append(running_total)
        self._total_centroids = running_total
        for entry in entries:
            logger.info(
                "Centroid summary for slide_id=%s wsi_path=%s ann_path=%s: %s",
                entry.slide_id,
                entry.wsi_path,
                entry.ann_path,
                f"count={self._centroid_count_by_slide.get(entry.slide_id, 0)} (disk-cached)",
            )
        total_centroids = sum(
            self._centroid_count_by_slide.get(entry.slide_id, 0) for entry in entries
        )
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

    def _build_centroid_cache_path(self, entry: SlideEntry) -> str:
        """Return deterministic disk-cache path for a slide centroid array."""
        digest = hashlib.sha1(
            f"{entry.slide_id}|{entry.ann_path}|{entry.wsi_path}".encode("utf-8")
        ).hexdigest()
        return str(self.centroid_cache_dir / f"{entry.slide_id}_{digest[:12]}.npy")

    def _get_slide_centroids_array(self, slide_id: str) -> np.memmap:
        """Load slide centroid array as mmap and keep an LRU of open arrays."""
        if slide_id in self._centroid_mem_cache:
            mm = self._centroid_mem_cache.pop(slide_id)
            self._centroid_mem_cache[slide_id] = mm
            return mm

        cache_path = self._centroid_cache_path_by_slide.get(slide_id)
        assert cache_path is not None, (
            f"Missing centroid cache for slide_id={slide_id}."
        )
        centroid_mm = np.load(cache_path, mmap_mode="r")
        assert centroid_mm.ndim == 2 and centroid_mm.shape[1] == 2, (
            "Centroid cache mmap must have shape [N, 2]."
        )
        self._centroid_mem_cache[slide_id] = centroid_mm

        while len(self._centroid_mem_cache) > self.in_memory_centroid_limit:
            self._centroid_mem_cache.popitem(last=False)
        return centroid_mm

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
        local_slide_ids = {entry.slide_id for entry in entries}
        local_cum_counts: List[int] = []
        local_entries: List[SlideEntry] = []
        running_total = 0
        for entry in self.valid_entries:
            if entry.slide_id not in local_slide_ids:
                continue
            running_total += self._centroid_count_by_slide[entry.slide_id]
            local_cum_counts.append(running_total)
            local_entries.append(entry)
        assert local_entries, "Expected at least one local entry for current rank."

        for _ in range(n_yield):
            global_idx = rng.randrange(running_total)
            slide_pos = bisect_right(local_cum_counts, global_idx)
            entry = local_entries[slide_pos]
            start = 0 if slide_pos == 0 else local_cum_counts[slide_pos - 1]
            center_idx = global_idx - start
            centroid_mm = self._get_slide_centroids_array(entry.slide_id)
            center = centroid_mm[center_idx]
            center_xy = (float(center[0]), float(center[1]))
            slide = self._get_open_slide(entry)
            patch = read_patch(slide, center_xy, self.big_size)

            assert patch.size == (self.big_size, self.big_size), (
                f"Expected patch size {(self.big_size, self.big_size)}, got {patch.size}."
            )
            img40 = self.rotcrop(patch)
            view1 = self.aug(img40)
            view2 = self.aug(img40)
            assert isinstance(view1, torch.Tensor), (
                "Augmentation pipeline must return torch.Tensor."
            )
            assert isinstance(view2, torch.Tensor), (
                "Augmentation pipeline must return torch.Tensor."
            )
            assert view1.shape == view2.shape, (
                "Paired views must have identical shapes."
            )
            assert view1.ndim == 3, (
                f"Expected CHW tensor, got shape {tuple(view1.shape)}."
            )
            expected_hw = (self.out_size, self.out_size)
            assert view1.shape[-2:] == expected_hw, (
                f"Expected HxW {expected_hw}, got {tuple(view1.shape[-2:])}."
            )
            if (
                self.debug_config is not None
                and self.debug_config.max_samples > 0
                and self._debug_count < self.debug_config.max_samples
                and worker_id == 0
                and rank == 0
            ):
                _save_debug_sample(
                    self.debug_config,
                    self._debug_count,
                    entry,
                    center_xy,
                    patch,
                    img40,
                    view1,
                    view2,
                    self.out_size,
                    self.big_size,
                )
                self._debug_count += 1
            yield view1, view2

    def _get_open_slide(self, entry: SlideEntry):
        """Get or open a worker-local OpenSlide handle with a tiny LRU cache."""
        if entry.slide_id in self._slide_cache:
            slide = self._slide_cache.pop(entry.slide_id)
            self._slide_cache[entry.slide_id] = slide
            return slide
        slide = safe_open_slide(entry.wsi_path)
        self._slide_cache[entry.slide_id] = slide
        while len(self._slide_cache) > self._slide_cache_limit:
            _, evicted_slide = self._slide_cache.popitem(last=False)
            evicted_slide.close()
        return slide

    def _close_slide_cache(self) -> None:
        """Close all cached slide handles and clear the worker-local cache."""
        while self._slide_cache:
            _, slide = self._slide_cache.popitem(last=False)
            try:
                slide.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup only.
                continue

    def __del__(self) -> None:
        """Best-effort cleanup for OpenSlide handles held by this iterable."""
        self._close_slide_cache()


class CellDataModule(PLDataModule):
    """Lightning DataModule for nucleus crop sampling."""

    def __init__(
        self,
        entries: List[SlideEntry],
        epoch_length: int,
        batch_size: int,
        num_workers: int,
        seed: int,
        out_size: int = 40,
        big_size: int = 60,
        debug_config: Optional[DebugSampleConfig] = None,
        centroid_cache_dir: Optional[str] = None,
        in_memory_centroid_limit: int = 8,
    ) -> None:
        super().__init__()
        assert isinstance(entries, list), (
            "entries must be a list of SlideEntry objects."
        )
        assert all(isinstance(entry, SlideEntry) for entry in entries), (
            "entries must contain SlideEntry objects."
        )
        assert epoch_length > 0, "epoch_length must be a positive integer."
        assert batch_size > 0, "batch_size must be a positive integer."
        assert num_workers >= 0, "num_workers must be non-negative."
        self.entries = entries
        self.epoch_length = epoch_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.out_size = out_size
        self.big_size = big_size
        self.debug_config = debug_config
        self.centroid_cache_dir = centroid_cache_dir
        self.in_memory_centroid_limit = in_memory_centroid_limit

    def train_dataloader(self) -> DataLoader:
        dataset = WSICellMoCoIterable(
            self.entries,
            self.epoch_length,
            seed=self.seed,
            out_size=self.out_size,
            big_size=self.big_size,
            debug_config=self.debug_config,
            centroid_cache_dir=self.centroid_cache_dir,
            in_memory_centroid_limit=self.in_memory_centroid_limit,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )
