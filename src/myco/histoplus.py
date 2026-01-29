"""Utilities for converting HistoPLUS cell masks to slide-level centroids."""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _deepzoom_tile_origin_and_scale(
    dz,
    tile_col: int,
    tile_row: int,
    dz_level: int,
) -> Optional[Tuple[float, float, float]]:
    """Resolve DeepZoom tile origin and scale factor to level-0 coordinates.

    Returns
    -------
    Optional[Tuple[float, float, float]]
        (tile_l0_x, tile_l0_y, level_scale) or None if the tile lookup fails.
    """
    try:
        (tile_l0_x, tile_l0_y), _, _ = dz.get_tile_coordinates(dz_level, (tile_col, tile_row))
    except Exception as exc:  # noqa: BLE001 - external OpenSlide failures are surfaced as warnings.
        logger.warning(
            "Could not get tile coordinates for tile (%d, %d) at level %d: %s",
            tile_col,
            tile_row,
            dz_level,
            exc,
        )
        return None

    level_scale = 2 ** (dz.level_count - 1 - dz_level)
    return float(tile_l0_x), float(tile_l0_y), float(level_scale)


def _coerce_point(point: object) -> Optional[Tuple[float, float]]:
    """Coerce a point-like object into an (x, y) tuple."""
    if point is None:
        return None
    if isinstance(point, dict):
        x_val = point.get("x") if "x" in point else point.get("X")
        y_val = point.get("y") if "y" in point else point.get("Y")
        if x_val is None or y_val is None:
            return None
        return float(x_val), float(y_val)
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    return None


def _iter_global_centroids(
    cell_masks: List[dict],
    dz,
    *,
    offset_x: int,
    offset_y: int,
    apply_bounds_offset: bool,
    progress: bool,
) -> Iterable[Tuple[float, float]]:
    """Yield slide-level centroids from HistoPLUS cell mask tiles.

    If a mask lacks an explicit centroid, the mean of its polygon coordinates is used.
    """
    iterator = cell_masks
    if progress:
        iterator = tqdm(iterator, desc="Remapping HistoPLUS centroids", unit="tile")

    def _centroid_from_coordinates(
        coords: Iterable[Iterable[float]],
    ) -> Optional[Tuple[float, float]]:
        coord_list = list(coords)
        if not coord_list:
            return None
        xs: List[float] = []
        ys: List[float] = []
        for coord in coord_list:
            pair = _coerce_point(coord)
            if pair is None:
                continue
            xs.append(pair[0])
            ys.append(pair[1])
        if not xs or not ys:
            return None
        assert len(xs) == len(ys), "Coordinate pairs must have matching x/y lengths."
        return sum(xs) / len(xs), sum(ys) / len(ys)

    for item in iterator:
        if not isinstance(item, dict):
            continue
        tile_col = int(item.get("x", 0))
        tile_row = int(item.get("y", 0))
        dz_level = int(item.get("level", 0))

        tile_info = _deepzoom_tile_origin_and_scale(dz, tile_col, tile_row, dz_level)
        if tile_info is None:
            continue
        tile_l0_x, tile_l0_y, level_scale = tile_info

        mask_payload = item.get("masks")
        if mask_payload is None:
            mask_payload = item.get("cell_masks") or item.get("cells") or item.get("objects")
        if not isinstance(mask_payload, list):
            continue
        for mask in mask_payload:
            if not isinstance(mask, dict):
                continue
            centroid = _coerce_point(mask.get("centroid"))
            if centroid is None:
                centroid = _centroid_from_coordinates(mask.get("coordinates", []))
            if centroid is None:
                continue
            local_x, local_y = centroid
            assert isinstance(local_x, float), "Centroid x must be numeric."
            assert isinstance(local_y, float), "Centroid y must be numeric."
            global_x = tile_l0_x + local_x * level_scale
            global_y = tile_l0_y + local_y * level_scale
            if apply_bounds_offset:
                global_x -= offset_x
                global_y -= offset_y
            yield global_x, global_y


def histoplus_centroids_from_payload(
    *,
    cell_masks: List[dict],
    slide_path: str,
    tile_size: Optional[int] = None,
    overlap: int = 0,
    apply_bounds_offset: bool = False,
    progress: bool = False,
) -> List[Tuple[float, float]]:
    """Convert HistoPLUS cell masks to slide-level centroids.

    Parameters
    ----------
    cell_masks : list[dict]
        Parsed HistoPLUS cell mask payload. Each entry contains tile metadata and masks.
    slide_path : str
        Path to the WSI used for coordinate conversion.
    tile_size : int, optional
        Tile size used for HistoPLUS inference. Defaults to the first tile width.
    overlap : int
        Tile overlap used during HistoPLUS inference.
    apply_bounds_offset : bool
        Whether to subtract OpenSlide bounds offsets. Keep False to align with ASAP XML.
    progress : bool
        Whether to show a tqdm progress bar.
    """
    if not cell_masks:
        return []

    assert isinstance(cell_masks, list), "cell_masks must be a list of dictionaries."
    if tile_size is None:
        tile_size = int(cell_masks[0].get("width", 224))
    assert tile_size > 0, "tile_size must be positive."
    assert overlap >= 0, "overlap must be non-negative."

    import openslide
    from openslide.deepzoom import DeepZoomGenerator

    slide = openslide.OpenSlide(slide_path)
    try:
        dz = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=False)
        offset_x = int(slide.properties.get("openslide.bounds-x", 0))
        offset_y = int(slide.properties.get("openslide.bounds-y", 0))
        centroids = list(
            _iter_global_centroids(
                cell_masks,
                dz,
                offset_x=offset_x,
                offset_y=offset_y,
                apply_bounds_offset=apply_bounds_offset,
                progress=progress,
            )
        )
    finally:
        slide.close()

    logger.info("Converted %d HistoPLUS centroids to slide coordinates.", len(centroids))
    return centroids
