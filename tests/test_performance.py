import time

from myco.performance import (
    IoTiming,
    get_cpu_rss_mb,
    get_gpu_memory_mb,
    measure_io_latency,
)


def test_measure_io_latency_returns_stats() -> None:
    def noop():
        time.sleep(0.001)

    timing = measure_io_latency(noop, samples=5)
    assert isinstance(timing, IoTiming)
    assert timing.samples == 5
    assert timing.mean_sec > 0
    assert timing.p95_sec > 0


def test_memory_helpers_return_numbers() -> None:
    assert get_cpu_rss_mb() > 0
    assert get_gpu_memory_mb() >= 0
