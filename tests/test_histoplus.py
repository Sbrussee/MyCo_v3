import json
from pathlib import Path

import pytest

from myco.data import load_centroids
from myco.histoplus import histoplus_centroids_from_payload


def test_load_centroids_histoplus_json_requires_slide_path(tmp_path: Path) -> None:
    json_path = tmp_path / "histoplus.json"
    json_path.write_text(json.dumps({"cell_masks": []}))

    with pytest.raises(ValueError, match="slide_path must be provided"):
        load_centroids(str(json_path))


def test_load_centroids_histoplus_json_uses_converter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    json_path = tmp_path / "histoplus.json"
    json_path.write_text(json.dumps({"cell_masks": [{"width": 224, "masks": []}]}))

    captured = {}

    def fake_converter(*, cell_masks, slide_path, apply_bounds_offset, progress):
        captured["cell_masks"] = cell_masks
        captured["slide_path"] = slide_path
        captured["apply_bounds_offset"] = apply_bounds_offset
        captured["progress"] = progress
        return [(1.0, 2.0)]

    monkeypatch.setattr("myco.histoplus.histoplus_centroids_from_payload", fake_converter)

    centroids = load_centroids(str(json_path), slide_path="slide.svs")
    assert centroids == [(1.0, 2.0)]
    assert captured["slide_path"] == "slide.svs"
    assert captured["apply_bounds_offset"] is False
    assert captured["progress"] is False


def test_histoplus_centroids_from_payload_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    openslide = pytest.importorskip("openslide")

    class FakeSlide:
        properties = {"openslide.bounds-x": "0", "openslide.bounds-y": "0"}

        def close(self) -> None:
            return None

    class FakeDeepZoomGenerator:
        def __init__(self, slide, tile_size: int, overlap: int, limit_bounds: bool) -> None:
            del slide, overlap, limit_bounds
            self.tile_size = tile_size
            self.level_count = 3

        def get_tile_coordinates(self, level: int, address: tuple[int, int]):
            tile_col, tile_row = address
            level_scale = 2 ** (self.level_count - 1 - level)
            return ((tile_col * self.tile_size * level_scale, tile_row * self.tile_size * level_scale), None, None)

    monkeypatch.setattr(openslide, "OpenSlide", lambda _: FakeSlide())
    monkeypatch.setattr(openslide.deepzoom, "DeepZoomGenerator", FakeDeepZoomGenerator)

    cell_masks = [
        {
            "level": 1,
            "x": 1,
            "y": 2,
            "width": 100,
            "masks": [
                {"centroid": [10, 20]},
                {"centroid": [30, 40]},
            ],
        }
    ]

    centroids = histoplus_centroids_from_payload(cell_masks=cell_masks, slide_path="fake.svs")
    assert centroids == [(220.0, 440.0), (260.0, 480.0)]
