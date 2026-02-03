"""End-to-end pipeline check for centroid parsing and sampling."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myco.data import CellDataModule, SlideEntry, build_entries_from_dirs  # noqa: E402
from myco.performance import log_performance_stats, measure_io_latency  # noqa: E402


class DummySlide:
    """Minimal slide stub that mimics OpenSlide read_region behavior."""

    def read_region(self, _loc, _level, size):
        return Image.new("RGB", size, color=(128, 128, 128))

    def close(self) -> None:
        return None


def run_pipeline_check() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        wsi_dir = tmp_path / "wsis"
        ann_dir = tmp_path / "anns"
        wsi_dir.mkdir()
        ann_dir.mkdir()

        (wsi_dir / "slide_1.svs").write_text("fake")
        (ann_dir / "slide_1.json").write_text(
            json.dumps({"centroids": [[15, 20], [30, 40]]})
        )

        entries = build_entries_from_dirs(str(wsi_dir), str(ann_dir))
        assert entries and isinstance(entries[0], SlideEntry)

        import myco.data as data_mod

        data_mod.safe_open_slide = lambda _path: DummySlide()

        datamodule = CellDataModule(
            entries=entries, epoch_length=2, batch_size=2, num_workers=0, seed=0
        )
        loader = datamodule.train_dataloader()
        batch = next(iter(loader))
        view1, view2 = batch

        assert view1.shape == view2.shape
        assert view1.shape[0] == 2
        assert view1.shape[1:] == (3, 40, 40)

        timing = measure_io_latency(
            lambda: DummySlide().read_region((0, 0), 0, (60, 60)), samples=5
        )
        log_performance_stats("pipeline_check", timing)

        logger.info("Pipeline check completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MyCo pipeline check.")
    _ = parser.parse_args()
    run_pipeline_check()


if __name__ == "__main__":
    main()
