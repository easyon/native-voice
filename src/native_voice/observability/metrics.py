"""Prometheus metrics. Labels are deliberately low-cardinality."""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


class Metrics:
    def __init__(self) -> None:
        self.ws_connections = Gauge("nv_ws_connections", "Open WebSocket connections", ["task"])
        self.tasks_total = Counter("nv_tasks_total", "Finished tasks", ["task", "outcome"])
        self.task_seconds = Histogram(
            "nv_task_duration_seconds", "Task wall time", ["task"], buckets=(0.5, 1, 2, 5, 10, 30, 60, 300)
        )
        self.tts_first_chunk = Histogram(
            "nv_tts_first_chunk_seconds",
            "TTS request to first audio frame",
            buckets=(0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 5.0),
        )
        self.tts_rtf = Histogram(
            "nv_tts_rtf", "TTS wall time / audio time", buckets=(0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0)
        )
        self.asr_partial = Histogram(
            "nv_asr_partial_seconds", "ASR partial interval", buckets=(0.3, 0.6, 0.9, 1.2, 2.0, 5.0)
        )
        self.errors_total = Counter("nv_errors_total", "Protocol/task errors", ["code"])
        self.quota_rejections = Counter("nv_quota_rejections_total", "Rejected by admission", ["reason"])
        self.audio_seconds = Counter("nv_audio_seconds_total", "Processed audio seconds", ["task"])
        self.tts_characters = Counter("nv_tts_characters_total", "Synthesized characters")
        self.tool_calls = Counter("nv_tool_calls_total", "Conversation tool calls", ["tool", "outcome"])
        self.tool_seconds = Histogram(
            "nv_tool_duration_seconds",
            "Tool handler wall time",
            ["tool"],
            buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
        )
        self.ready = Gauge("nv_ready", "Service readiness (1=ready)")


METRICS = Metrics()


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
