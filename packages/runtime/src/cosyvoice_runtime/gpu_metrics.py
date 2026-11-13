"""GPU utilization gauges for ``/metrics``.

The runtime already records what it can see from Python (queue wait, TTFF,
RTF, active streams). GPU SM utilization is the missing signal for two
questions:

* Is a replica compute-bound? High utilization plus flat aggregate throughput
  means add replicas, not threads.
* Should a horizontal autoscaler add a replica? ``cosyvoice_active_streams``
  rises when callers queue behind a saturated GPU, but utilization shows it
  earlier and survives traffic that has not yet timed out.

Sampling is deliberately out of process: ``nvidia-smi`` is present whenever
the driver is, and no CUDA context or extra Python binding is needed. A
5-second cadence costs one short-lived subprocess per interval, which is
noise next to the model. Values are exported as ratios (0..1) so PromQL and
the Prometheus adapter can average them directly.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from collections.abc import Callable

from .metrics import Metrics

logger = logging.getLogger("cosyvoice_runtime.gpu_metrics")

_QUERY = "utilization.gpu,memory.used,memory.total"

GpuSample = tuple[float, int, int]
"""``(utilization_ratio, memory_used_bytes, memory_total_bytes)``."""


def parse_gpu_sample(stdout: str) -> GpuSample | None:
    """Parse one ``nvidia-smi --query-gpu`` CSV line.

    Returns ``None`` for empty, multi-GPU (the caller always passes ``-i``),
    ``[N/A]`` (MIG without utilization telemetry) or malformed output instead
    of raising: a metrics sampler must never take the gateway down.
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        # More than one line means ``-i`` did not select a single device
        # (multi-GPU or MIG child devices). Guessing which one to publish
        # would make the gauge ambiguous for autoscaling, so publish nothing.
        return None
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 3:
        return None
    try:
        utilization = float(parts[0])
        used_mib = int(float(parts[1]))
        total_mib = int(float(parts[2]))
    except ValueError:
        return None
    if not 0.0 <= utilization <= 100.0 or used_mib < 0 or total_mib <= 0:
        return None
    return utilization / 100.0, used_mib * 2**20, total_mib * 2**20


def read_gpu_sample(device: int = 0, timeout: float = 2.0) -> GpuSample | None:
    """Run one ``nvidia-smi`` query. Never raises."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "-i",
                str(device),
                f"--query-gpu={_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return parse_gpu_sample(completed.stdout)


class GpuSampler:
    """Publish GPU gauges into :class:`Metrics` from a daemon thread.

    The sampler is best-effort by design: a missing driver, a MIG slice
    without utilization telemetry or a wedged ``nvidia-smi`` only means the
    ``cosyvoice_gpu_*`` series are absent or stale, never that startup fails.
    ``sample_fn`` is injectable for tests.
    """

    def __init__(
        self,
        metrics: Metrics,
        interval_s: float = 5.0,
        device: int = 0,
        sample_fn: Callable[[], GpuSample | None] | None = None,
    ) -> None:
        self.metrics = metrics
        self.interval_s = max(0.0, float(interval_s))
        self.device = device
        self._sample_fn = sample_fn
        self._stop = threading.Event()
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name="gpu-metrics",
            daemon=True,
        )

    def start(self) -> None:
        if self._started:
            return
        if self.interval_s <= 0:
            logger.debug("GPU metric sampling disabled")
            return
        if self._sample_fn is None:
            if shutil.which("nvidia-smi") is None:
                logger.debug("nvidia-smi not found; GPU gauges are not exported")
                return
        self._started = True
        self._thread.start()

    def collect(self) -> bool:
        """Sample once and update gauges. Returns whether a sample was used."""
        sample = self._sample()
        if sample is None:
            return False
        utilization, used_bytes, total_bytes = sample
        self.metrics.set_gpu(utilization, used_bytes, total_bytes)
        return True

    def stop(self, timeout: float = 2.0) -> None:
        if not self._thread.is_alive():
            return
        self._stop.set()
        self._thread.join(timeout=timeout)

    def _sample(self) -> GpuSample | None:
        if self._sample_fn is not None:
            return self._sample_fn()
        return read_gpu_sample(self.device)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.collect()
            except Exception:  # noqa: BLE001 - a sampler must never break /metrics
                logger.debug("GPU sample failed", exc_info=True)
            self._stop.wait(self.interval_s)
