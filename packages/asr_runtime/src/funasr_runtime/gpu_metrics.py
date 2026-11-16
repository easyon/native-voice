from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Callable

from .metrics import Metrics

GpuSample = tuple[float, int, int]


def parse_gpu_sample(stdout: str) -> GpuSample | None:
    line = next((part.strip() for part in stdout.splitlines() if part.strip()), "")
    if not line:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        utilization = float(parts[0])
        used = int(float(parts[1]))
        total = int(float(parts[2]))
    except ValueError:
        return None
    return utilization / 100.0, used * 1024 * 1024, total * 1024 * 1024


def read_gpu_sample(device: int = 0, timeout: float = 2.0) -> GpuSample | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    command = [
        exe,
        f"--id={device}",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return parse_gpu_sample(proc.stdout)


class GpuSampler:
    def __init__(
        self,
        metrics: Metrics,
        interval_s: float = 5.0,
        device: int = 0,
        sample_fn: Callable[[], GpuSample | None] | None = None,
    ) -> None:
        self._metrics = metrics
        self._interval = max(0.0, float(interval_s))
        self._device = max(0, int(device))
        self._sample_fn = sample_fn or (lambda: read_gpu_sample(self._device))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._interval <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="funasr-gpu-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.collect()

    def collect(self) -> bool:
        sample = self._sample_fn()
        if sample is None:
            return False
        utilization, used, total = sample
        self._metrics.set_gpu(utilization, used, total)
        return True
