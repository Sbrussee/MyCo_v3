"""Performance utilities for memory and IO monitoring."""

from __future__ import annotations

import logging
import resource
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IoTiming:
    """Summary of IO timing statistics.

    Attributes
    ----------
    mean_sec : float
        Mean wall-clock time per IO operation in seconds.
    p95_sec : float
        95th percentile time per IO operation in seconds.
    samples : int
        Number of IO samples collected.
    """

    mean_sec: float
    p95_sec: float
    samples: int


def get_cpu_rss_mb() -> float:
    """Return the current process RSS in MB."""
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss_kb) / 1024.0


def get_gpu_memory_mb(device: Optional[torch.device] = None) -> float:
    """Return allocated GPU memory in MB for the given device."""
    if not torch.cuda.is_available():
        return 0.0
    dev = device if device is not None else torch.device("cuda")
    return float(torch.cuda.memory_allocated(dev)) / 1e6


def measure_io_latency(read_fn: Callable[[], None], samples: int = 20) -> IoTiming:
    """Measure IO latency for a callable that performs one IO operation."""
    if samples <= 0:
        raise ValueError("samples must be positive.")
    durations = []
    for _ in range(samples):
        start = time.perf_counter()
        read_fn()
        durations.append(time.perf_counter() - start)
    durations_sorted = sorted(durations)
    idx = max(0, int(round(0.95 * (samples - 1))))
    p95 = durations_sorted[idx]
    mean = sum(durations) / samples
    return IoTiming(mean_sec=float(mean), p95_sec=float(p95), samples=samples)


def log_performance_stats(tag: str, timing: IoTiming) -> None:
    """Log IO timing and memory stats."""
    logger.info(
        "%s IO timing: mean=%.4fs p95=%.4fs samples=%d",
        tag,
        timing.mean_sec,
        timing.p95_sec,
        timing.samples,
    )
    logger.info("%s CPU RSS: %.1f MB", tag, get_cpu_rss_mb())
    if torch.cuda.is_available():
        logger.info("%s GPU allocated: %.1f MB", tag, get_gpu_memory_mb())
