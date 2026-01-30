"""Annotation parsing and coordinate remapping utilities."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlideGeometry:
    """Slide-level geometry and coordinate metadata.

    Attributes
    ----------
    width : int
        Level-0 width in pixels.
    height : int
        Level-0 height in pixels.
    level_downsamples : tuple[float, ...]
        Downsample factors for each OpenSlide level (level-0 is 1.0).
    bounds_x : int
        X offset of the valid bounds rectangle in level-0 coordinates.
    bounds_y : int
        Y offset of the valid bounds rectangle in level-0 coordinates.
    bounds_width : int
        Width of the valid bounds rectangle in level-0 coordinates.
    bounds_height : int
        Height of the valid bounds rectangle in level-0 coordinates.
    mpp_x : float | None
        Microns-per-pixel in X, if present in slide properties.
    mpp_y : float | None
        Microns-per-pixel in Y, if present in slide properties.
    """

    width: int
    height: int
    level_downsamples: Tuple[float, ...]
    bounds_x: int
    bounds_y: int
    bounds_width: int
    bounds_height: int
    mpp_x: Optional[float]
    mpp_y: Optional[float]

    def contains(self, x: float, y: float) -> bool:
        """Return True if ``(x, y)`` lies within the slide bounds."""
        return (
            self.bounds_x <= x < self.bounds_x + self.bounds_width
            and self.bounds_y <= y < self.bounds_y + self.bounds_height
        )


@dataclass(frozen=True)
class CoordinateTransform:
    """Affine transform for mapping coordinates to level-0 space."""

    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    source: str


def get_slide_geometry(slide_path: str) -> SlideGeometry:
    """Read slide geometry metadata using OpenSlide."""
    import openslide

    slide = openslide.OpenSlide(slide_path)
    try:
        width, height = slide.dimensions
        level_downsamples = tuple(float(value) for value in slide.level_downsamples)
        assert width > 0 and height > 0, "Slide dimensions must be positive."
        assert level_downsamples, "Slide must expose level_downsamples."
        bounds_x = int(slide.properties.get("openslide.bounds-x", 0))
        bounds_y = int(slide.properties.get("openslide.bounds-y", 0))
        bounds_width = int(slide.properties.get("openslide.bounds-width", width))
        bounds_height = int(slide.properties.get("openslide.bounds-height", height))
        mpp_x = slide.properties.get("openslide.mpp-x")
        mpp_y = slide.properties.get("openslide.mpp-y")
        mpp_x_val = float(mpp_x) if mpp_x is not None else None
        mpp_y_val = float(mpp_y) if mpp_y is not None else None
    finally:
        slide.close()
    return SlideGeometry(
        width=int(width),
        height=int(height),
        level_downsamples=level_downsamples,
        bounds_x=bounds_x,
        bounds_y=bounds_y,
        bounds_width=bounds_width,
        bounds_height=bounds_height,
        mpp_x=mpp_x_val,
        mpp_y=mpp_y_val,
    )


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


def _coerce_centroid(item: object) -> Optional[Tuple[float, float]]:
    if item is None:
        return None
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return float(item[0]), float(item[1])
    if isinstance(item, dict):
        if "centroid" in item:
            return _coerce_centroid(item.get("centroid"))
        if "center" in item:
            return _coerce_centroid(item.get("center"))
        if "x" in item and "y" in item:
            return float(item["x"]), float(item["y"])
        if "X" in item and "Y" in item:
            return float(item["X"]), float(item["Y"])
    return None


def _append_centroids(coords: List[Tuple[float, float]], items: Iterable[object]) -> None:
    for item in items:
        centroid = _coerce_centroid(item)
        if centroid is None and isinstance(item, dict):
            centroid = _coerce_centroid(item.get("centroid") or item.get("center"))
        if centroid is None:
            continue
        coords.append((float(centroid[0]), float(centroid[1])))


def parse_geojson_centroids_from_payload(data: Dict[str, object]) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    if not isinstance(data, dict):
        return coords

    features = data.get("features", [])
    if not isinstance(features, list):
        return coords

    try:
        from shapely.geometry import shape
    except Exception:  # noqa: BLE001 - shapely is optional.
        shape = None

    for feat in features:
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry")
        if geom is None:
            continue
        if shape is not None:
            geom_obj = shape(geom)
            centroid = geom_obj if geom_obj.geom_type == "Point" else geom_obj.centroid
            coords.append((float(centroid.x), float(centroid.y)))
        else:
            if geom.get("type") == "Point":
                point = geom.get("coordinates")
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    coords.append((float(point[0]), float(point[1])))
            else:
                logger.warning("Shapely unavailable; skipping non-point GeoJSON geometry.")
    return coords


def parse_json_centroids_from_payload(data: object) -> List[Tuple[float, float]]:
    """Parse centroids from JSON payloads that are not GeoJSON."""
    coords: List[Tuple[float, float]] = []

    if isinstance(data, dict):
        if "features" in data:
            return parse_geojson_centroids_from_payload(data)
        if "centroids" in data:
            centroids = data.get("centroids", [])
            if isinstance(centroids, list):
                _append_centroids(coords, centroids)
        for key in (
            "cells",
            "objects",
            "instances",
            "annotations",
            "nuclei",
            "points",
            "detections",
            "regions",
            "items",
        ):
            items = data.get(key)
            if isinstance(items, list):
                _append_centroids(coords, items)
        return coords

    if isinstance(data, list):
        _append_centroids(coords, data)
        return coords

    return coords


def _summarize_centroids(centroids: List[Tuple[float, float]]) -> str:
    if not centroids:
        return "count=0"
    xs = [coord[0] for coord in centroids]
    ys = [coord[1] for coord in centroids]
    return (
        f"count={len(centroids)} x_range=({min(xs):.2f},{max(xs):.2f}) "
        f"y_range=({min(ys):.2f},{max(ys):.2f})"
    )


def _extract_metadata_dict(data: Dict[str, object]) -> Dict[str, object]:
    for key in ("metadata", "meta", "info", "header"):
        meta = data.get(key)
        if isinstance(meta, dict):
            return meta
    return {}


def _extract_transform_from_metadata(
    data: Dict[str, object],
    slide_geometry: Optional[SlideGeometry],
) -> CoordinateTransform:
    meta = _extract_metadata_dict(data)
    lookup = {**data, **meta}

    def _maybe_float(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    scale_x = _maybe_float(lookup.get("scale_x"))
    scale_y = _maybe_float(lookup.get("scale_y"))
    if scale_x is None or scale_y is None:
        scale = _maybe_float(lookup.get("scale"))
        if scale is not None:
            scale_x = scale if scale_x is None else scale_x
            scale_y = scale if scale_y is None else scale_y

    downsample = _maybe_float(lookup.get("downsample"))
    level = lookup.get("level") or lookup.get("slide_level") or lookup.get("level_index")
    level_idx = int(level) if isinstance(level, (int, float, str)) and str(level).isdigit() else None

    if scale_x is None or scale_y is None:
        if downsample is not None:
            scale_x = downsample if scale_x is None else scale_x
            scale_y = downsample if scale_y is None else scale_y

    if (scale_x is None or scale_y is None) and level_idx is not None and slide_geometry is not None:
        if 0 <= level_idx < len(slide_geometry.level_downsamples):
            ds = slide_geometry.level_downsamples[level_idx]
            scale_x = ds if scale_x is None else scale_x
            scale_y = ds if scale_y is None else scale_y

    if scale_x is None:
        scale_x = 1.0
    if scale_y is None:
        scale_y = 1.0

    offset_x = _maybe_float(lookup.get("offset_x"))
    offset_y = _maybe_float(lookup.get("offset_y"))
    if offset_x is None or offset_y is None:
        offset = lookup.get("offset") or lookup.get("origin") or lookup.get("bounds")
        if isinstance(offset, (list, tuple)) and len(offset) >= 2:
            if offset_x is None:
                offset_x = float(offset[0])
            if offset_y is None:
                offset_y = float(offset[1])
        if isinstance(offset, dict):
            if offset_x is None and "x" in offset:
                offset_x = float(offset["x"])
            if offset_y is None and "y" in offset:
                offset_y = float(offset["y"])

    if offset_x is None:
        offset_x = 0.0
    if offset_y is None:
        offset_y = 0.0

    source = "metadata"
    return CoordinateTransform(
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=offset_x,
        offset_y=offset_y,
        source=source,
    )


def apply_coordinate_transform(
    centroids: List[Tuple[float, float]],
    transform: CoordinateTransform,
) -> List[Tuple[float, float]]:
    """Apply a coordinate transform to centroid coordinates."""
    if not centroids:
        return []
    import numpy as np

    coords = np.asarray(centroids, dtype=np.float64)
    assert coords.ndim == 2 and coords.shape[1] == 2, "Centroids must be shaped (N, 2)."
    scaled = coords * np.array([transform.scale_x, transform.scale_y], dtype=np.float64)
    shifted = scaled + np.array([transform.offset_x, transform.offset_y], dtype=np.float64)
    return [(float(x), float(y)) for x, y in shifted]


def filter_centroids_to_bounds(
    centroids: List[Tuple[float, float]],
    geometry: SlideGeometry,
) -> List[Tuple[float, float]]:
    """Filter centroids to those within slide bounds."""
    if not centroids:
        return []
    filtered = [(x, y) for x, y in centroids if geometry.contains(x, y)]
    if len(filtered) != len(centroids):
        logger.info(
            "Filtered centroids to slide bounds (%d/%d remain).",
            len(filtered),
            len(centroids),
        )
    return filtered


def parse_geojson_centroids(path: str) -> List[Tuple[float, float]]:
    """Parse centroids from a GeoJSON or JSON annotation file."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return parse_geojson_centroids_from_payload(data)


def read_annotation_text(path: str) -> str:
    """Read annotation file text for debugging."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def summarize_annotation_payload(data: object) -> str:
    """Summarize the top-level annotation payload for debugging."""
    if isinstance(data, dict):
        keys = sorted(list(data.keys()))
        keys_preview = ", ".join(keys[:10])
        suffix = "" if len(keys) <= 10 else f" (+{len(keys) - 10} more)"
        return f"dict keys: [{keys_preview}]{suffix}"
    if isinstance(data, list):
        preview = data[0] if data else None
        preview_type = type(preview).__name__
        return f"list length={len(data)} first_type={preview_type}"
    return f"type={type(data).__name__}"


def truncate_payload(payload: str, limit: int = 50000) -> str:
    """Truncate payload strings to avoid excessive logging."""
    assert limit > 0, "limit must be positive."
    if len(payload) <= limit:
        return payload
    return f"{payload[:limit]}\n... [truncated {len(payload) - limit} chars]"


def is_histoplus_payload(data: object) -> bool:
    if isinstance(data, dict):
        return "cell_masks" in data or "cellMasks" in data
    if isinstance(data, list):
        return any(
            isinstance(item, dict) and ("masks" in item or "cell_masks" in item or "cells" in item)
            for item in data
        )
    return False


def parse_json_with_remap(
    *,
    data: object,
    slide_path: Optional[str],
) -> List[Tuple[float, float]]:
    """Parse JSON centroids and remap to slide coordinates if metadata exists."""
    centroids = parse_json_centroids_from_payload(data)
    if not centroids:
        return []
    if isinstance(data, dict):
        geometry = None
        if slide_path:
            try:
                geometry = get_slide_geometry(slide_path)
            except Exception as exc:  # noqa: BLE001 - OpenSlide may be unavailable in tests.
                logger.warning("Failed to load slide geometry for remapping: %s", exc)
        transform = _extract_transform_from_metadata(data, geometry)
        if transform.scale_x != 1.0 or transform.scale_y != 1.0 or transform.offset_x != 0.0 or transform.offset_y != 0.0:
            centroids = apply_coordinate_transform(centroids, transform)
            logger.info(
                "Applied coordinate transform (%s): scale=(%.3f, %.3f) offset=(%.1f, %.1f).",
                transform.source,
                transform.scale_x,
                transform.scale_y,
                transform.offset_x,
                transform.offset_y,
            )
    return centroids


def load_centroids_from_json(
    *,
    path: str,
    data: object,
    slide_path: Optional[str],
    progress: bool,
) -> List[Tuple[float, float]]:
    """Load centroids from JSON payloads, remapping when required."""
    if is_histoplus_payload(data):
        if slide_path is None:
            raise ValueError("slide_path must be provided for HistoPLUS JSON annotations.")
        from .histoplus import histoplus_centroids_from_payload

        if isinstance(data, dict):
            cell_masks = data.get("cell_masks") or data.get("cellMasks") or []
        else:
            cell_masks = data
        assert isinstance(cell_masks, list), "cell_masks must be a list for HistoPLUS conversion."
        return histoplus_centroids_from_payload(
            cell_masks=cell_masks,
            slide_path=slide_path,
            apply_bounds_offset=bool(
                isinstance(data, dict) and data.get("apply_bounds_offset")
            ),
            progress=progress,
        )

    return parse_json_with_remap(data=data, slide_path=slide_path)
