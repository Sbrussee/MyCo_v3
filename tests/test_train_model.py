from pathlib import Path
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_model  # noqa: E402
from myco.callbacks import BatchMetricsLogger  # noqa: E402


def test_build_logger_uses_csv(tmp_path):
    logger = train_model.build_logger(str(tmp_path))
    assert isinstance(logger, CSVLogger)
    assert logger.save_dir == str(tmp_path)
    assert logger.name == "logs"


def test_build_callbacks_includes_logging_and_checkpointing(tmp_path):
    eval_cb = pl.Callback()
    callbacks = train_model.build_callbacks(
        str(tmp_path), eval_cb, log_every_n_batches=10
    )
    callback_types = {type(cb) for cb in callbacks}

    assert eval_cb in callbacks
    assert ModelCheckpoint in callback_types
    assert LearningRateMonitor in callback_types
    assert DeviceStatsMonitor in callback_types
    assert TQDMProgressBar in callback_types
    assert BatchMetricsLogger in callback_types


def test_resolve_resume_checkpoint_prefers_latest_versioned_last_checkpoint(
    tmp_path,
) -> None:
    moco_ckpt = tmp_path / "moco-001.ckpt"
    moco_ckpt.write_text("placeholder")
    last_ckpt = tmp_path / "last.ckpt"
    last_ckpt.write_text("placeholder")
    (tmp_path / "last-v4.ckpt").write_text("placeholder")
    last_v5_ckpt = tmp_path / "last-v5.ckpt"
    last_v5_ckpt.write_text("placeholder")

    resolved = train_model.resolve_resume_checkpoint(str(tmp_path))
    assert resolved == str(last_v5_ckpt)


def test_resolve_resume_checkpoint_falls_back_to_unversioned_last(tmp_path) -> None:
    (tmp_path / "moco-001.ckpt").write_text("placeholder")
    last_ckpt = tmp_path / "last.ckpt"
    last_ckpt.write_text("placeholder")

    resolved = train_model.resolve_resume_checkpoint(str(tmp_path))
    assert resolved == str(last_ckpt)


def test_resolve_resume_checkpoint_falls_back_to_latest_epoch(tmp_path) -> None:
    (tmp_path / "moco-001.ckpt").write_text("placeholder")
    latest_ckpt = tmp_path / "moco-010.ckpt"
    latest_ckpt.write_text("placeholder")

    resolved = train_model.resolve_resume_checkpoint(str(tmp_path))
    assert resolved == str(latest_ckpt)
