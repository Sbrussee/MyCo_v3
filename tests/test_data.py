import json
import pytest
from pathlib import Path

import torch
from PIL import Image

from myco.data import (
    CellDataModule,
    DebugSampleConfig,
    SlideEntry,
    WSICellMoCoIterable,
    _save_debug_sample,
    build_entries_from_dirs,
    load_centroids,
    parse_xml_centroids,
    read_slide_labels,
)


def test_read_slide_labels_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("slide_id,label\nslide_1,MF\nslide_2,BID\n")
    labels = read_slide_labels(str(csv_path))
    assert labels == {"slide_1": 1, "slide_2": 0}


def test_read_slide_labels_csv_slide_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("slide,label\nslide_a,1\nslide_b,0\n")
    labels = read_slide_labels(str(csv_path))
    assert labels == {"slide_a": 1, "slide_b": 0}


def test_read_slide_labels_json(tmp_path: Path) -> None:
    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps({"slide_1": "MF", "slide_2": 0}))
    labels = read_slide_labels(str(json_path))
    assert labels == {"slide_1": 1, "slide_2": 0}


def test_parse_xml_centroids(tmp_path: Path) -> None:
    pytest.importorskip("lxml")
    xml_path = tmp_path / "ann.xml"
    xml_path.write_text(
        """
        <Annotations>
          <Annotation>
            <Coordinates>
              <Coordinate X=\"10\" Y=\"20\" />
              <Coordinate X=\"30\" Y=\"40\" />
            </Coordinates>
          </Annotation>
        </Annotations>
        """
    )
    coords = parse_xml_centroids(str(xml_path))
    assert coords == [(10.0, 20.0), (30.0, 40.0)]


def test_load_centroids_geojson(tmp_path: Path) -> None:
    pytest.importorskip("shapely")
    geojson_path = tmp_path / "ann.geojson"
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [12, 34]},
                        "properties": {},
                    }
                ],
            }
        )
    )
    coords = load_centroids(str(geojson_path))
    assert coords == [(12.0, 34.0)]


def test_load_centroids_json_centroids_key(tmp_path: Path) -> None:
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps({"centroids": [[1, 2], [3.5, 4.5]]}))
    coords = load_centroids(str(json_path))
    assert coords == [(1.0, 2.0), (3.5, 4.5)]


def test_load_centroids_json_list_of_dicts(tmp_path: Path) -> None:
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps([{"centroid": [10, 20]}, {"x": 30, "y": 40}]))
    coords = load_centroids(str(json_path))
    assert coords == [(10.0, 20.0), (30.0, 40.0)]


def test_load_centroids_json_points_key(tmp_path: Path) -> None:
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps({"points": [{"x": 5, "y": 6}, {"centroid": [7, 8]}]}))
    coords = load_centroids(str(json_path))
    assert coords == [(5.0, 6.0), (7.0, 8.0)]


def test_load_centroids_json_detections_key(tmp_path: Path) -> None:
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps({"detections": [{"x": 11, "y": 12}]}))
    coords = load_centroids(str(json_path))
    assert coords == [(11.0, 12.0)]


def test_build_entries_from_dirs(tmp_path: Path) -> None:
    wsi_dir = tmp_path / "wsis"
    ann_dir = tmp_path / "anns"
    wsi_dir.mkdir()
    ann_dir.mkdir()

    (wsi_dir / "slide_1.svs").write_text("fake")
    (ann_dir / "slide_1.xml").write_text("<Annotations />")

    entries = build_entries_from_dirs(str(wsi_dir), str(ann_dir))
    assert len(entries) == 1
    assert entries[0].slide_id == "slide_1"


def test_cell_datamodule_inherits_lightning_datamodule() -> None:
    pl = pytest.importorskip("pytorch_lightning")
    entries = [SlideEntry(slide_id="slide_1", wsi_path="slide_1.svs", ann_path="slide_1.xml")]
    datamodule = CellDataModule(entries=entries, epoch_length=1, batch_size=1, num_workers=0, seed=0)

    assert isinstance(datamodule, pl.LightningDataModule)
    assert hasattr(datamodule, "on_exception")


def test_wsi_iterable_filters_empty_centroids(monkeypatch) -> None:
    entries = [
        SlideEntry(slide_id="slide_1", wsi_path="slide_1.svs", ann_path="slide_1.xml"),
        SlideEntry(slide_id="slide_2", wsi_path="slide_2.svs", ann_path="slide_2.xml"),
    ]

    def fake_load_centroids(path: str, slide_path: str | None = None):
        del slide_path
        if "slide_1" in path:
            return [(1.0, 2.0)]
        return []

    monkeypatch.setattr("myco.data.load_centroids", fake_load_centroids)

    dataset = WSICellMoCoIterable(entries=entries, epoch_length=10, seed=0)
    assert len(dataset.valid_entries) == 1
    assert dataset.valid_entries[0].slide_id == "slide_1"


def test_wsi_iterable_raises_without_centroids(monkeypatch) -> None:
    entries = [SlideEntry(slide_id="slide_1", wsi_path="slide_1.svs", ann_path="slide_1.xml")]

    def fake_load_centroids(path: str, slide_path: str | None = None):
        del slide_path
        return []

    monkeypatch.setattr("myco.data.load_centroids", fake_load_centroids)

    with pytest.raises(ValueError, match="No valid entries with centroids"):
        WSICellMoCoIterable(entries=entries, epoch_length=10, seed=0)


def test_save_debug_sample(tmp_path: Path) -> None:
    config = DebugSampleConfig(output_dir=tmp_path, max_samples=1)
    entry = SlideEntry(slide_id="slide_1", wsi_path="slide_1.svs", ann_path="slide_1.xml")
    patch = Image.new("RGB", (60, 60), color=(255, 0, 0))
    view1 = torch.zeros((3, 40, 40), dtype=torch.float32)
    view2 = torch.ones((3, 40, 40), dtype=torch.float32)

    _save_debug_sample(
        config=config,
        sample_idx=0,
        entry=entry,
        center=(10.0, 20.0),
        patch=patch,
        view1=view1,
        view2=view2,
        out_size=40,
        big_size=60,
    )

    assert (tmp_path / "slide_1_sample_0000_patch.png").exists()
    assert (tmp_path / "slide_1_sample_0000_view1.png").exists()
    assert (tmp_path / "slide_1_sample_0000_view2.png").exists()
    assert (tmp_path / "slide_1_sample_0000_meta.json").exists()


def test_pipeline_smoke_with_dummy_slide(tmp_path: Path, monkeypatch) -> None:
    wsi_dir = tmp_path / "wsis"
    ann_dir = tmp_path / "anns"
    wsi_dir.mkdir()
    ann_dir.mkdir()

    (wsi_dir / "slide_1.svs").write_text("fake")
    (ann_dir / "slide_1.json").write_text(json.dumps({"centroids": [[15, 20], [30, 40]]}))

    entries = build_entries_from_dirs(str(wsi_dir), str(ann_dir))

    class DummySlide:
        def read_region(self, _loc, _level, size):
            return Image.new("RGB", size, color=(128, 128, 128))

        def close(self):
            return None

    monkeypatch.setattr("myco.data.safe_open_slide", lambda _path: DummySlide())

    datamodule = CellDataModule(entries=entries, epoch_length=2, batch_size=2, num_workers=0, seed=0)
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    view1, view2 = batch

    assert view1.shape == view2.shape
    assert view1.shape[0] == 2
    assert view1.shape[1:] == (3, 40, 40)
