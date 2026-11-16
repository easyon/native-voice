from __future__ import annotations

import io
import json
import math
import struct
import wave
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from funasr_runtime.main import create_app
from funasr_runtime.settings import RuntimeConfig

PATH = "/api-ws/v1/inference"
TASK_ID = "2bf83b9a-baeb-4fda-8d9a-000000000001"


@pytest.fixture
def config() -> RuntimeConfig:
    config = RuntimeConfig.default()
    config.engine.mock.partial_interval_ms = 100
    config.engine.mock.segment_ms = 1_000
    config.engine.mock.latency_ms = 0
    return config


@pytest.fixture
def client(config: RuntimeConfig) -> Iterator[TestClient]:
    with TestClient(create_app(config)) as test_client:
        yield test_client


def tone_pcm(duration_ms: int, *, sample_rate: int = 16_000, amplitude: int = 8_000) -> bytes:
    samples = int(sample_rate * duration_ms / 1000)
    return b"".join(
        struct.pack(
            "<h",
            int(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate)),
        )
        for index in range(samples)
    )


def silence_pcm(duration_ms: int, *, sample_rate: int = 16_000) -> bytes:
    return b"\x00\x00" * int(sample_rate * duration_ms / 1000)


def wav_bytes(pcm: bytes, *, sample_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


def run_task(model: str = "fun-asr-realtime", **parameters: object) -> dict:
    params = {
        "format": "pcm",
        "sample_rate": 16_000,
        "max_sentence_silence": 1_300,
    }
    params.update(parameters)
    return {
        "header": {
            "action": "run-task",
            "task_id": TASK_ID,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": model,
            "parameters": params,
            "input": {},
        },
    }


def finish_task(task_id: str = TASK_ID) -> dict:
    return {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }


def receive_events(websocket, *, limit: int = 500) -> list[dict]:
    events: list[dict] = []
    for _ in range(limit):
        message = websocket.receive()
        text = message.get("text")
        if text is None:
            continue
        event = json.loads(text)
        events.append(event)
        if event["header"]["event"] in {"task-finished", "task-failed"}:
            break
    return events


def sentence_events(events: list[dict]) -> list[dict]:
    return [
        event["payload"]["output"]["sentence"]
        for event in events
        if event["header"]["event"] == "result-generated"
    ]
