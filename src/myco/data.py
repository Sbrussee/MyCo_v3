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

from .annotations import (
    filter_centroids_to_bounds,
    get_slide_geometry,
    load_centroids_from_json,
    parse_geojson_centroids,
    parse_geojson_centroids_from_payload,
    parse_json_centroids_from_payload,
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
    if _centroid_cache_is_fresh(cache_path, path):
        return _read_centroid_cache(cache_path)

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

    _write_centroid_cache(cache_path, centroids)
    logger.info(
        "Parsed centroids from %s (format=%s, %s).",
        path,
        Path(path).suffix,
        _summarize_centroids(centroids),
    )
    return centroids


def _centroid_cache_path(path: str) -> Path:
    """Return the cache path used to store serialized centroids for an annotation."""
    ann_path = Path(path)
    return ann_path.with_suffix(ann_path.suffix + ".centroids.json")


def _centroid_cache_is_fresh(cache_path: Path, ann_path: str) -> bool:
    """Return True if a centroid cache exists and is newer than the annotation file."""
    if not cache_path.exists():
        return False
    try:
        cache_mtime = cache_path.stat().st_mtime
        ann_mtime = Path(ann_path).stat().st_mtime
    except OSError:
        return False
    return cache_mtime >= ann_mtime


def _read_centroid_cache(path: Path) -> List[Tuple[float, float]]:
    """Read cached centroid coordinates from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        cached = json.load(handle)
    centroids = [(float(item[0]), float(item[1])) for item in cached]
    logger.info("Loaded cached centroids from %s (%s).", path, _summarize_centroids(centroids))
    return centroids


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


def _save_debug_sample(
    config: DebugSampleConfig,
    sample_idx: int,
    entry: SlideEntry,
    center: Tuple[float, float],
    patch,
    view1,
    view2,
    out_size: int,
    big_size: int,
) -> None:
    """Save debug images + metadata for a sampled patch."""
    import torch
    from torchvision.transforms.functional import to_pil_image

    assert config.max_samples >= 0, "max_samples must be non-negative."
    assert patch.size == (big_size, big_size), f"Expected patch size {(big_size, big_size)}."
    assert isinstance(view1, torch.Tensor), "view1 must be a torch.Tensor."
    assert isinstance(view2, torch.Tensor), "view2 must be a torch.Tensor."
    assert view1.shape == view2.shape, "Debug views must match shapes."
    assert view1.shape[-2:] == (out_size, out_size), "Unexpected debug view size."

    safe_slide_id = entry.slide_id.replace(os.sep, "_")
    sample_prefix = f"{safe_slide_id}_sample_{sample_idx:04d}"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    patch_path = config.output_dir / f"{sample_prefix}_patch.png"
    view1_path = config.output_dir / f"{sample_prefix}_view1.png"
    view2_path = config.output_dir / f"{sample_prefix}_view2.png"
    metadata_path = config.output_dir / f"{sample_prefix}_meta.json"

    patch.save(patch_path)
    to_pil_image(view1.detach().cpu().clamp(0, 1)).save(view1_path)
    to_pil_image(view2.detach().cpu().clamp(0, 1)).save(view2_path)

    metadata = {
        "slide_id": entry.slide_id,
        "wsi_path": entry.wsi_path,
        "ann_path": entry.ann_path,
        "center": [float(center[0]), float(center[1])],
        "patch_size": [int(big_size), int(big_size)],
        "view_shape": list(view1.shape),
        "patch_path": str(patch_path),
        "view1_path": str(view1_path),
        "view2_path": str(view2_path),
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
        entries.append(SlideEntry(slide_id=stem, wsi_path=str(wsi_path), ann_path=str(ann_path)))

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
    ) -> None:
        super().__init__()
        self.all_entries = entries
        self.epoch_length = epoch_length
        self.seed = seed
        self.out_size = out_size
        self.big_size = big_size
        self.debug_config = debug_config
        self._debug_count = 0

        from .augment import RotationCrop40, build_lemon_a1_gray_transform

        self.rotcrop = RotationCrop40(big_size=big_size, out_size=out_size, degrees=360.0)
        self.aug = build_lemon_a1_gray_transform(img_size=out_size)

        self.centroids: Dict[str, List[Tuple[float, float]]] = {}
        try:
            from tqdm.auto import tqdm

            entry_iter = tqdm(entries, desc="Loading centroids", unit="slide")
        except Exception:  # noqa: BLE001 - tqdm is optional at runtime.
            entry_iter = entries
        for entry in entry_iter:
            self.centroids[entry.slide_id] = load_centroids(entry.ann_path, slide_path=entry.wsi_path)
        self.valid_entries = [entry for entry in entries if self.centroids.get(entry.slide_id)]
        for entry in entries:
            logger.info(
                "Centroid summary for slide_id=%s wsi_path=%s ann_path=%s: %s",
                entry.slide_id,
                entry.wsi_path,
                entry.ann_path,
                _summarize_centroids(self.centroids.get(entry.slide_id, [])),
            )
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

            assert patch.size == (self.big_size, self.big_size), (
                f"Expected patch size {(self.big_size, self.big_size)}, got {patch.size}."
            )
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
                    center,
                    patch,
                    view1,
                    view2,
                    self.out_size,
                    self.big_size,
                )
                self._debug_count += 1
            yield view1, view2


class CellDataModule(PLDataModule):
    """Lightning DataModule for nucleus crop sampling."""

    def __init__(
        self,
        entries: List[SlideEntry],
        epoch_length: int,
        batch_size: int,
        num_workers: int,
        seed: int,
        debug_config: Optional[DebugSampleConfig] = None,
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
        self.debug_config = debug_config

    def train_dataloader(self) -> DataLoader:
        dataset = WSICellMoCoIterable(
            self.entries,
            self.epoch_length,
            seed=self.seed,
            debug_config=self.debug_config,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
