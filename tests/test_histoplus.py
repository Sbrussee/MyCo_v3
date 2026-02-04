import json
from pathlib import Path

import pytest

from myco.data import load_centroids
from myco.histoplus import histoplus_centroids_from_payload


def test_load_centroids_histoplus_json_assumes_global_without_tile_metadata(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "histoplus.json"
    json_path.write_text(json.dumps({"cell_masks": []}))

    centroids = load_centroids(str(json_path))
    assert centroids == []


def test_load_centroids_histoplus_json_uses_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "histoplus.json"
    json_path.write_text(
        json.dumps(
            {"cell_masks": [{"x": 0, "y": 0, "level": 0, "width": 224, "masks": []}]}
        )
    )

    captured = {}

    def fake_converter(*, cell_masks, slide_path, apply_bounds_offset, progress):
        captured["cell_masks"] = cell_masks
        captured["slide_path"] = slide_path
        captured["apply_bounds_offset"] = apply_bounds_offset
        captured["progress"] = progress
        return [(1.0, 2.0)]

    monkeypatch.setattr(
        "myco.histoplus.histoplus_centroids_from_payload", fake_converter
    )

    centroids = load_centroids(str(json_path), slide_path="slide.svs")
    assert centroids == [(1.0, 2.0)]
    assert captured["slide_path"] == "slide.svs"
    assert captured["apply_bounds_offset"] is False
    assert captured["progress"] is False


def test_load_centroids_histoplus_json_requires_slide_path_with_tile_metadata(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "histoplus.json"
    json_path.write_text(
        json.dumps({"cell_masks": [{"x": 0, "y": 0, "level": 0, "masks": []}]})
    )

    with pytest.raises(ValueError, match="slide_path must be provided"):
        load_centroids(str(json_path))


def test_load_centroids_histoplus_json_with_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "histoplus.json"
    json_path.write_text(
        json.dumps(
            {
                "model_name": "histoplus_v1",
                "inference_mpp": 0.25,
                "cell_masks": [
                    {"width": 224, "height": 224, "masks": [{"centroid": [1, 2]}]}
                ],
            }
        )
    )

    def fake_converter(*, cell_masks, slide_path, apply_bounds_offset, progress):
        raise AssertionError("histoplus_centroids_from_payload should not be called")

    monkeypatch.setattr(
        "myco.histoplus.histoplus_centroids_from_payload", fake_converter
    )

    centroids = load_centroids(str(json_path), slide_path="slide.svs")
    assert centroids == [(1.0, 2.0)]


def test_load_centroids_histoplus_real_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    openslide = pytest.importorskip("openslide")

    class FakeSlide:
        properties = {"openslide.bounds-x": "0", "openslide.bounds-y": "0"}
        level_downsamples = [float(2**i) for i in range(15)]

        def close(self) -> None:
            return None

    class FakeDeepZoomGenerator:
        def __init__(
            self, slide, tile_size: int, overlap: int, limit_bounds: bool
        ) -> None:
            del slide, overlap, limit_bounds
            self.tile_size = tile_size
            self.level_count = 15

        def get_tile_coordinates(self, level: int, address: tuple[int, int]):
            tile_col, tile_row = address
            level_scale = 2 ** (self.level_count - 1 - level)
            return (
                (
                    tile_col * self.tile_size * level_scale,
                    tile_row * self.tile_size * level_scale,
                ),
                None,
                None,
            )

    monkeypatch.setattr(openslide, "OpenSlide", lambda _: FakeSlide())
    monkeypatch.setattr(openslide.deepzoom, "DeepZoomGenerator", FakeDeepZoomGenerator)

    payload = {
        "model_name": "histoplus_cellvit_segmentor_20x",
        "inference_mpp": 0.5,
        "cell_masks": [
            {
                "level": 14.0,
                "x": 28.0,
                "y": 25.0,
                "width": 224.0,
                "height": 224.0,
                "masks": [
                    {
                        "cell_id": 0.0,
                        "cell_type": "Cancer cell",
                        "confidence": 0.9999999843750003,
                        "coordinates": [
                            [123.0, 7.0],
                            [120.0, 10.0],
                        ],
                        "centroid": [123.0, 12.0],
                    },
                    {
                        "cell_id": 1.0,
                        "cell_type": "Cancer cell",
                        "confidence": 0.9999999879518074,
                        "coordinates": [
                            [158.0, 7.0],
                            [156.0, 9.0],
                        ],
                        "centroid": [160.0, 11.0],
                    },
                    {
                        "cell_id": 2.0,
                        "cell_type": "Epithelial",
                        "confidence": 0.8999999850000003,
                        "coordinates": [
                            [-2.0, 10.0],
                            [0.0, 10.0],
                        ],
                        "centroid": [0.0, 14.0],
                    },
                ],
            }
        ],
    }

    json_path = tmp_path / "histoplus.json"
    json_path.write_text(json.dumps(payload))

    centroids = load_centroids(str(json_path), slide_path="slide.svs")
    assert centroids == [(6395.0, 5612.0), (6432.0, 5611.0), (6272.0, 5614.0)]


def test_load_centroids_histoplus_json_skips_remap_for_global_coordinate_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "histoplus.json"
    json_path.write_text(
        json.dumps(
            {
                "coordinate_space": "global",
                "cell_masks": [{"masks": [{"centroid": [10, 20]}]}],
            }
        )
    )

    def fake_converter(*, cell_masks, slide_path, apply_bounds_offset, progress):
        raise AssertionError("histoplus_centroids_from_payload should not be called")

    monkeypatch.setattr(
        "myco.histoplus.histoplus_centroids_from_payload", fake_converter
    )

    centroids = load_centroids(str(json_path))
    assert centroids == [(10.0, 20.0)]


def test_histoplus_centroids_from_payload_basic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openslide = pytest.importorskip("openslide")

    class FakeSlide:
        properties = {"openslide.bounds-x": "0", "openslide.bounds-y": "0"}
        level_downsamples = [1.0, 2.0, 4.0]

        def close(self) -> None:
            return None

    class FakeDeepZoomGenerator:
        def __init__(
            self, slide, tile_size: int, overlap: int, limit_bounds: bool
        ) -> None:
            del slide, overlap, limit_bounds
            self.tile_size = tile_size
            self.level_count = 3

        def get_tile_coordinates(self, level: int, address: tuple[int, int]):
            tile_col, tile_row = address
            level_scale = 2 ** (self.level_count - 1 - level)
            return (
                (
                    tile_col * self.tile_size * level_scale,
                    tile_row * self.tile_size * level_scale,
                ),
                None,
                None,
            )

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

    centroids = histoplus_centroids_from_payload(
        cell_masks=cell_masks, slide_path="fake.svs"
    )
    assert centroids == [(220.0, 440.0), (260.0, 480.0)]


def test_histoplus_centroids_from_payload_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openslide = pytest.importorskip("openslide")

    class FakeSlide:
        properties = {"openslide.bounds-x": "0", "openslide.bounds-y": "0"}
        level_downsamples = [1.0, 2.0]

        def close(self) -> None:
            return None

    class FakeDeepZoomGenerator:
        def __init__(
            self, slide, tile_size: int, overlap: int, limit_bounds: bool
        ) -> None:
            del slide, overlap, limit_bounds
            self.tile_size = tile_size
            self.level_count = 2

        def get_tile_coordinates(self, level: int, address: tuple[int, int]):
            tile_col, tile_row = address
            level_scale = 2 ** (self.level_count - 1 - level)
            return (
                (
                    tile_col * self.tile_size * level_scale,
                    tile_row * self.tile_size * level_scale,
                ),
                None,
                None,
            )

    monkeypatch.setattr(openslide, "OpenSlide", lambda _: FakeSlide())
    monkeypatch.setattr(openslide.deepzoom, "DeepZoomGenerator", FakeDeepZoomGenerator)

    cell_masks = [
        {
            "level": 0,
            "x": 0,
            "y": 0,
            "width": 100,
            "masks": [
                {
                    "coordinates": [
                        [10, 20],
                        [30, 40],
                        [50, 60],
                    ]
                }
            ],
        }
    ]

    centroids = histoplus_centroids_from_payload(
        cell_masks=cell_masks, slide_path="fake.svs"
    )
    assert centroids == [(30.0, 40.0)]


def test_histoplus_centroids_supports_dict_centroids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openslide = pytest.importorskip("openslide")

    class FakeSlide:
        properties = {"openslide.bounds-x": "0", "openslide.bounds-y": "0"}
        level_downsamples = [1.0, 2.0]

        def close(self) -> None:
            return None

    class FakeDeepZoomGenerator:
        def __init__(
            self, slide, tile_size: int, overlap: int, limit_bounds: bool
        ) -> None:
            del slide, overlap, limit_bounds
            self.tile_size = tile_size
            self.level_count = 2

        def get_tile_coordinates(self, level: int, address: tuple[int, int]):
            tile_col, tile_row = address
            level_scale = 2 ** (self.level_count - 1 - level)
            return (
                (
                    tile_col * self.tile_size * level_scale,
                    tile_row * self.tile_size * level_scale,
                ),
                None,
                None,
            )

    monkeypatch.setattr(openslide, "OpenSlide", lambda _: FakeSlide())
    monkeypatch.setattr(openslide.deepzoom, "DeepZoomGenerator", FakeDeepZoomGenerator)

    cell_masks = [
        {
            "level": 0,
            "x": 0,
            "y": 0,
            "width": 100,
            "masks": [
                {"centroid": {"x": 12, "y": 34}},
            ],
        }
    ]

    centroids = histoplus_centroids_from_payload(
        cell_masks=cell_masks, slide_path="fake.svs"
    )
    assert centroids == [(12.0, 34.0)]


def test_load_centroids_histoplus_list_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "histoplus.json"
    json_path.write_text(json.dumps([{"width": 224, "masks": [{"centroid": [1, 2]}]}]))

    def fake_converter(*, cell_masks, slide_path, apply_bounds_offset, progress):
        return [(3.0, 4.0)]

    monkeypatch.setattr(
        "myco.histoplus.histoplus_centroids_from_payload", fake_converter
    )

    centroids = load_centroids(str(json_path), slide_path="slide.svs")
    assert centroids == [(1.0, 2.0)]


def test_histoplus_centroids_from_payload_pixel_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openslide = pytest.importorskip("openslide")

    class FakeSlide:
        properties = {"openslide.bounds-x": "0", "openslide.bounds-y": "0"}
        level_downsamples = [1.0, 4.0]

        def close(self) -> None:
            return None

    class FakeDeepZoomGenerator:
        def __init__(
            self, slide, tile_size: int, overlap: int, limit_bounds: bool
        ) -> None:
            del slide, overlap, limit_bounds, tile_size
            self.level_count = 2

        def get_tile_coordinates(self, level: int, address: tuple[int, int]):
            raise ValueError("Simulated DeepZoom failure")

    monkeypatch.setattr(openslide, "OpenSlide", lambda _: FakeSlide())
    monkeypatch.setattr(openslide.deepzoom, "DeepZoomGenerator", FakeDeepZoomGenerator)

    cell_masks = [
        {
            "level": 1,
            "x": 100,
            "y": 200,
            "width": 224,
            "masks": [
                {"centroid": [10, 20]},
            ],
        }
    ]

    centroids = histoplus_centroids_from_payload(
        cell_masks=cell_masks, slide_path="fake.svs"
    )
    assert centroids == [(440.0, 880.0)]


def test_histoplus_centroids_from_payload_out_of_bounds_indices_use_pixel_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openslide = pytest.importorskip("openslide")

    class FakeSlide:
        properties = {"openslide.bounds-x": "0", "openslide.bounds-y": "0"}
        level_downsamples = [1.0, 4.0]

        def close(self) -> None:
            return None

    class FakeDeepZoomGenerator:
        def __init__(
            self, slide, tile_size: int, overlap: int, limit_bounds: bool
        ) -> None:
            del slide, overlap, limit_bounds
            self.tile_size = tile_size
            self.level_count = 2
            self.level_tiles = [(2, 2), (2, 2)]

        def get_tile_coordinates(self, level: int, address: tuple[int, int]):
            tile_col, tile_row = address
            level_scale = 2 ** (self.level_count - 1 - level)
            return (
                (
                    tile_col * self.tile_size * level_scale,
                    tile_row * self.tile_size * level_scale,
                ),
                None,
                None,
            )

    monkeypatch.setattr(openslide, "OpenSlide", lambda _: FakeSlide())
    monkeypatch.setattr(openslide.deepzoom, "DeepZoomGenerator", FakeDeepZoomGenerator)

    cell_masks = [
        {
            "level": 1,
            "x": 224,
            "y": 448,
            "width": 224,
            "masks": [
                {"centroid": [10, 20]},
            ],
        }
    ]

    centroids = histoplus_centroids_from_payload(
        cell_masks=cell_masks, slide_path="fake.svs"
    )
    assert centroids == [(896.0, 1792.0)]
