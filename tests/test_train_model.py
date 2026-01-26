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

import train_model
from myco.callbacks import BatchMetricsLogger


def test_build_logger_uses_csv(tmp_path):
    logger = train_model.build_logger(str(tmp_path))
    assert isinstance(logger, CSVLogger)
    assert logger.save_dir == str(tmp_path)
    assert logger.name == "logs"


def test_build_callbacks_includes_logging_and_checkpointing(tmp_path):
    eval_cb = pl.Callback()
    callbacks = train_model.build_callbacks(str(tmp_path), eval_cb, log_every_n_batches=10)
    callback_types = {type(cb) for cb in callbacks}

    assert eval_cb in callbacks
    assert ModelCheckpoint in callback_types
    assert LearningRateMonitor in callback_types
    assert DeviceStatsMonitor in callback_types
    assert TQDMProgressBar in callback_types
    assert BatchMetricsLogger in callback_types
