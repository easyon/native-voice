from __future__ import annotations

import base64
import json

from fastapi.testclient import TestClient
from tts_runtime.main import create_app
from tts_runtime.settings import RuntimeConfig


def _config(chunk_bytes: int) -> RuntimeConfig:
    config = RuntimeConfig.default()
    config.storage.type = "none"
    config.engine.mock.chunk_ms = 1_000
    config.limits.stream_chunk_bytes = chunk_bytes
    return config


def _sse_audio(config: RuntimeConfig, fmt: str = "pcm") -> list[bytes]:
    body = {
        "model": "my-tts-prod",
        "input": {"text": "你好。今天好吗？", "voice": "voice-a", "format": fmt},
    }
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/api/v1/services/audio/tts/SpeechSynthesizer",
            json=body,
            headers={"X-DashScope-SSE": "enable"},
        )
    assert response.status_code == 200, response.text
    chunks: list[bytes] = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        output = event.get("output") or {}
        if output.get("type") != "sentence-synthesis":
            continue
        data = (output.get("audio") or {}).get("data") or ""
        if data:
            chunks.append(base64.b64decode(data))
    return chunks


def test_sse_splits_long_engine_chunks() -> None:
    unsplit = _sse_audio(_config(0))
    split = _sse_audio(_config(16_000))

    # The mock engine emits one 1-second chunk per sentence; with a 16 KB
    # frame limit each chunk is cut into multiple SSE audio events.
    assert len(split) > len(unsplit)
    assert all(len(chunk) <= 16_000 for chunk in split)
    assert sum(len(chunk) for chunk in split) == sum(len(chunk) for chunk in unsplit)


def test_wav_stream_header_only_on_first_frame() -> None:
    chunks = _sse_audio(_config(16_000), fmt="wav")

    assert len(chunks) > 1
    assert chunks[0].startswith(b"RIFF")
    assert b"".join(chunks).startswith(b"RIFF")
    assert not chunks[1].startswith(b"RIFF")
