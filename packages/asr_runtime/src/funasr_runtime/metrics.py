from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)


def _labels(**labels: str) -> str:
    if not labels:
        return ""
    body = ",".join(
        f'{key}="{_escape(value)}"' for key, value in sorted(labels.items())
    )
    return "{" + body + "}"


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _value(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


@dataclass
class _Histogram:
    counts: list[int] = field(default_factory=lambda: [0] * len(_BUCKETS))
    total: float = 0.0
    count: int = 0

    def observe(self, value: float) -> None:
        value = max(0.0, float(value))
        for index, bound in enumerate(_BUCKETS):
            if value <= bound:
                self.counts[index] += 1
                break
        self.total += value
        self.count += 1


class Metrics:
    """Dependency-free Prometheus exposition for the ASR gateway."""

    def __init__(self, service_name: str = "funasr-runtime", version: str = "") -> None:
        self._lock = threading.RLock()
        self._started = time.monotonic()
        self._service_name = service_name
        self._version = version
        self._counters: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
        self._counters[("funasr_requests_total", ())] = 0
        self._counters[("funasr_errors_total", ())] = 0
        self._counters[("funasr_audio_seconds_total", ())] = 0
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[
            tuple[str, tuple[tuple[str, str], ...]], _Histogram
        ] = {}
        self._active_streams = 0
        self._ws_connections = 0
        self._ready = 0

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        with self._lock:
            self._counters[key] += value

    def request(self, api: str, status: str) -> None:
        self.inc("funasr_requests_total", api=api, status=status)

    def error(self, code: str, api: str = "") -> None:
        if api:
            self.inc("funasr_errors_total", code=code, api=api)
        else:
            self.inc("funasr_errors_total", code=code)

    def sentence(self) -> None:
        self.inc("funasr_sentences_total")

    def audio_seconds(self, seconds: float) -> None:
        self.inc("funasr_audio_seconds_total", max(0.0, float(seconds)))

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        with self._lock:
            self._gauges[key] = float(value)

    def stream_started(self) -> None:
        with self._lock:
            self._active_streams += 1
            self._gauges[("funasr_active_streams", ())] = self._active_streams

    def stream_finished(self) -> None:
        with self._lock:
            self._active_streams = max(0, self._active_streams - 1)
            self._gauges[("funasr_active_streams", ())] = self._active_streams

    def ws_connected(self) -> None:
        with self._lock:
            self._ws_connections += 1
            self._gauges[("funasr_ws_connections", ())] = self._ws_connections

    def ws_disconnected(self) -> None:
        with self._lock:
            self._ws_connections = max(0, self._ws_connections - 1)
            self._gauges[("funasr_ws_connections", ())] = self._ws_connections

    def set_ready(self, ready: bool) -> None:
        with self._lock:
            self._ready = 1 if ready else 0
            self._gauges[("funasr_ready", ())] = self._ready

    def set_gpu(self, utilization: float, used_bytes: int, total_bytes: int) -> None:
        used = max(0, int(used_bytes))
        total = max(0, int(total_bytes))
        with self._lock:
            self._gauges[("funasr_gpu_utilization_ratio", ())] = max(
                0.0, min(1.0, float(utilization))
            )
            self._gauges[("funasr_gpu_memory_used_bytes", ())] = used
            self._gauges[("funasr_gpu_memory_total_bytes", ())] = total
            self._gauges[("funasr_gpu_memory_utilization_ratio", ())] = (
                used / total if total else 0.0
            )

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        with self._lock:
            histogram = self._histograms.get(key)
            if histogram is None:
                histogram = self._histograms[key] = _Histogram()
            histogram.observe(value)

    def request_duration(self, seconds: float, **labels: str) -> None:
        self.observe("funasr_request_duration_seconds", seconds, **labels)

    def first_partial(self, seconds: float, **labels: str) -> None:
        self.observe("funasr_first_partial_seconds", seconds, **labels)

    def final_latency(self, seconds: float, **labels: str) -> None:
        self.observe("funasr_final_latency_seconds", seconds, **labels)

    def rtf(self, value: float, **labels: str) -> None:
        self.observe("funasr_rtf", value, **labels)

    def render(self) -> str:
        with self._lock:
            lines: list[str] = [
                "# HELP funasr_build_info Build information.",
                "# TYPE funasr_build_info gauge",
                "funasr_build_info"
                f'{{service="{_escape(self._service_name)}",'
                f'version="{_escape(self._version)}"}} 1',
                "# HELP funasr_uptime_seconds Process uptime.",
                "# TYPE funasr_uptime_seconds gauge",
                f"funasr_uptime_seconds {time.monotonic() - self._started:.3f}",
            ]

            counters: dict[str, list[tuple[str, float]]] = {}
            for (name, labels), value in sorted(self._counters.items()):
                counters.setdefault(name, []).append((_labels(**dict(labels)), value))
            for name, samples in sorted(counters.items()):
                lines.append(f"# TYPE {name} counter")
                for label_text, value in samples:
                    lines.append(f"{name}{label_text} {_value(value)}")

            gauges: dict[str, list[tuple[str, float]]] = {}
            for (name, labels), value in sorted(self._gauges.items()):
                gauges.setdefault(name, []).append((_labels(**dict(labels)), value))
            for name, samples in sorted(gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                for label_text, value in samples:
                    lines.append(f"{name}{label_text} {_value(value)}")

            histograms: dict[str, list[tuple[str, _Histogram]]] = {}
            for (name, labels), histogram in sorted(self._histograms.items()):
                histograms.setdefault(name, []).append(
                    (_labels(**dict(labels)), histogram)
                )
            for name, samples in sorted(histograms.items()):
                lines.append(f"# TYPE {name} histogram")
                for label_text, histogram in samples:
                    cumulative = 0
                    for index, bound in enumerate(_BUCKETS):
                        cumulative += histogram.counts[index]
                        labels = self._with_le(label_text, bound)
                        lines.append(f"{name}_bucket{labels} {cumulative}")
                    labels = self._with_le(label_text, "+Inf")
                    lines.append(f"{name}_bucket{labels} {histogram.count}")
                    lines.append(f"{name}_sum{label_text} {_value(histogram.total)}")
                    lines.append(f"{name}_count{label_text} {histogram.count}")
            lines.append("")
            return "\n".join(lines)

    @staticmethod
    def _with_le(label_text: str, bound: float | str) -> str:
        le = f"{bound:g}" if isinstance(bound, (int, float)) else bound
        if not label_text:
            return f'{{le="{le}"}}'
        return label_text[:-1] + f',le="{le}"}}'

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_streams": self._active_streams,
                "ws_connections": self._ws_connections,
                "ready": self._ready,
                "counters": dict(self._counters),
            }
