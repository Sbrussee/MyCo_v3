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
from myco.model import MoCoV3Lit  # noqa: E402


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


def test_is_resume_checkpoint_compatible_accepts_matching_state_dict(
    tmp_path,
) -> None:
    ckpt_path = tmp_path / "match.ckpt"
    state_dict = {"q_proj.net.0.weight": train_model.torch.ones((2, 2))}
    train_model.torch.save({"state_dict": state_dict}, ckpt_path)

    model = object.__new__(MoCoV3Lit)
    model.on_load_checkpoint = lambda checkpoint: None
    model.state_dict = lambda: {"q_proj.net.0.weight": train_model.torch.zeros((2, 2))}

    assert train_model.is_resume_checkpoint_compatible(model, str(ckpt_path))


def test_is_resume_checkpoint_compatible_rejects_missing_or_unexpected_keys(
    tmp_path,
) -> None:
    ckpt_path = tmp_path / "mismatch.ckpt"
    state_dict = {"q_proj.fc1.weight": train_model.torch.ones((2, 2))}
    train_model.torch.save({"state_dict": state_dict}, ckpt_path)

    model = object.__new__(MoCoV3Lit)
    model.state_dict = lambda: {
        "q_proj.net.0.weight": train_model.torch.zeros((2, 2)),
        "predictor.net.0.weight": train_model.torch.zeros((2, 2)),
    }

    def _normalize(checkpoint: dict[str, object]) -> None:
        checkpoint["state_dict"] = {
            "q_proj.net.0.weight": train_model.torch.ones((2, 2)),
            "extra.weight": train_model.torch.ones((2, 2)),
        }

    model.on_load_checkpoint = _normalize

    assert not train_model.is_resume_checkpoint_compatible(model, str(ckpt_path))
