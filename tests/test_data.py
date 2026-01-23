import json
import pytest
from pathlib import Path

from myco.data import build_entries_from_dirs, load_centroids, parse_xml_centroids, read_slide_labels


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
