import types

import torch

from myco.callbacks import BatchMetricsLogger, extract_batch_size


def test_extract_batch_size_validates_shapes():
    batch = (
        torch.zeros((4, 3, 224, 224)),
        torch.zeros((4, 3, 224, 224)),
    )
    assert extract_batch_size(batch) == 4


def test_batch_metrics_logger_logs_every_n_batches():
    callback = BatchMetricsLogger(log_every_n_batches=2)
    logger_calls = {}

    class DummyLogger:
        def log_metrics(self, metrics, step):
            logger_calls["metrics"] = metrics
            logger_calls["step"] = step

    dummy_trainer = types.SimpleNamespace(logger=DummyLogger(), global_step=10)
    dummy_module = types.SimpleNamespace(device=torch.device("cpu"))
    batch = (
        torch.zeros((2, 3, 224, 224)),
        torch.zeros((2, 3, 224, 224)),
    )

    callback.on_train_batch_start(dummy_trainer, dummy_module, batch, batch_idx=1)
    callback.on_train_batch_end(dummy_trainer, dummy_module, None, batch, batch_idx=1)

    assert "metrics" in logger_calls
    assert "cpu_mem_rss_mb" in logger_calls["metrics"]
