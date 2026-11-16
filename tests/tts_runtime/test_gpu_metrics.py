from __future__ import annotations

import pytest
from tts_runtime.gpu_metrics import GpuSampler, parse_gpu_sample, read_gpu_sample
from tts_runtime.metrics import Metrics
from tts_runtime.settings import RuntimeConfig


def test_parse_gpu_sample_converts_units() -> None:
    sample = parse_gpu_sample(" 42, 1234, 46068\n")
    assert sample is not None
    utilization, used, total = sample
    assert utilization == pytest.approx(0.42)
    assert used == 1234 * 2**20
    assert total == 46068 * 2**20


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "\n",
        "45, 1000\n",  # driver omitted memory.total
        "[N/A], 1000, 46068\n",  # MIG without utilization telemetry
        "101, 1000, 46068\n",  # out of range
        "45, 1000, 0\n",  # bogus total
        "45, 1000, 46068\n45, 2000, 46068\n",  # more than one device
    ],
)
def test_parse_gpu_sample_rejects_bad_output(stdout: str) -> None:
    assert parse_gpu_sample(stdout) is None


def test_metrics_expose_gpu_gauges() -> None:
    metrics = Metrics()
    metrics.set_gpu(0.75, 6 * 2**30, 48 * 2**30)
    text = metrics.render()
    assert "cosyvoice_gpu_utilization_ratio 0.75" in text
    assert f"cosyvoice_gpu_memory_used_bytes {6 * 2**30}" in text
    assert f"cosyvoice_gpu_memory_total_bytes {48 * 2**30}" in text
    assert "cosyvoice_gpu_memory_utilization_ratio 0.125" in text


def test_metrics_clamp_gpu_utilization_and_tolerate_zero_total() -> None:
    metrics = Metrics()
    metrics.set_gpu(1.7, 10, 0)
    text = metrics.render()
    assert "cosyvoice_gpu_utilization_ratio 1" in text
    assert "cosyvoice_gpu_memory_utilization_ratio 0" in text


def test_sampler_collects_into_metrics() -> None:
    metrics = Metrics()
    sampler = GpuSampler(metrics, interval_s=0, sample_fn=lambda: (0.5, 1024, 4096))
    assert sampler.collect() is True
    assert "cosyvoice_gpu_utilization_ratio 0.5" in metrics.render()


def test_sampler_skips_failed_samples() -> None:
    metrics = Metrics()
    sampler = GpuSampler(metrics, interval_s=0, sample_fn=lambda: None)
    assert sampler.collect() is False
    assert "cosyvoice_gpu_utilization_ratio" not in metrics.render()


def test_sampler_start_is_idempotent_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    def fake_sample() -> tuple[float, int, int]:
        calls.append(1.0)
        return 0.1, 1, 2

    metrics = Metrics()
    sampler = GpuSampler(metrics, interval_s=0.01, sample_fn=fake_sample)
    sampler.start()
    sampler.start()
    sampler.stop(timeout=2)
    assert calls


def test_sampler_without_nvidia_smi_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tts_runtime.gpu_metrics.shutil.which", lambda _: None)
    sampler = GpuSampler(Metrics(), interval_s=5)
    sampler.start()
    sampler.stop()
    assert "cosyvoice_gpu_utilization_ratio" not in sampler.metrics.render()


def test_read_gpu_sample_handles_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tts_runtime.gpu_metrics.shutil.which", lambda _: None)
    assert read_gpu_sample(0) is None


def test_config_defaults_enable_gpu_sampling() -> None:
    config = RuntimeConfig()
    assert config.metrics.gpu_sampling_interval_s == 5.0
    assert config.metrics.gpu_device == 0
