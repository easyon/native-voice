from __future__ import annotations

import base64
import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from tts_runtime.main import create_app
from tts_runtime.settings import RuntimeConfig


@pytest.fixture
def config() -> RuntimeConfig:
    config = RuntimeConfig.default()
    config.storage.type = "none"
    return config


@pytest.fixture
def client(config: RuntimeConfig) -> Iterator[TestClient]:
    with TestClient(create_app(config)) as test_client:
        yield test_client


def payload(text: str = "你好。今天想了解点什么？", **overrides):
    input_ = {
        "text": text,
        "voice": "voice-a",
        "format": "wav",
        "sample_rate": 24000,
    }
    input_.update(overrides)
    return {"model": "my-tts-prod", "input": input_}


def test_health_and_metrics(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["engine"] == "mock"
    assert "cosyvoice_requests_total" in client.get("/metrics").text


def test_http_non_stream_returns_audio(client: TestClient) -> None:
    response = client.post(
        "/api/v1/services/audio/tts/SpeechSynthesizer",
        json=payload("你好。"),
    )
    assert response.status_code == 200
    body = response.json()
    audio = base64.b64decode(body["output"]["audio"]["data"])
    assert audio.startswith(b"RIFF")
    assert body["output"]["finish_reason"] == "stop"
    assert body["usage"]["characters"] == 3
    assert response.headers.get("X-Request-Id")


def test_http_unknown_model_and_voice(client: TestClient) -> None:
    response = client.post(
        "/api/v1/services/audio/tts/SpeechSynthesizer",
        json={**payload(), "model": "missing"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "InvalidParameter"
    assert response.json()["request_id"]

    response = client.post(
        "/api/v1/services/audio/tts/SpeechSynthesizer",
        json=payload(voice="missing-voice"),
    )
    assert response.status_code == 400
    assert response.json()["code"] == "VoiceNotFound"


def test_http_auth() -> None:
    config = RuntimeConfig.default()
    config.storage.type = "none"
    config.auth.enabled = True
    config.auth.api_keys = ["test-key"]
    with TestClient(create_app(config)) as auth_client:
        unauth = auth_client.post(
            "/api/v1/services/audio/tts/SpeechSynthesizer",
            json=payload(),
        )
        assert unauth.status_code == 401
        assert unauth.json()["code"] == "Unauthorized"

        ok = auth_client.post(
            "/api/v1/services/audio/tts/SpeechSynthesizer",
            json=payload("你好。"),
            headers={"Authorization": "Bearer test-key"},
        )
        assert ok.status_code == 200


def test_sse_event_order(client: TestClient) -> None:
    response = client.post(
        "/api/v1/services/audio/tts/SpeechSynthesizer",
        json=payload("你好。今天好吗？"),
        headers={"X-DashScope-SSE": "enable"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    types = [
        event["output"]["type"]
        for event in events
        if event.get("output", {}).get("type")
    ]
    assert types == [
        "sentence-begin",
        "sentence-synthesis",
        "sentence-synthesis",
        "sentence-end",
        "sentence-begin",
        "sentence-synthesis",
        "sentence-synthesis",
        "sentence-end",
    ]
    assert events[-1]["output"]["finish_reason"] == "stop"
    assert events[-1]["usage"]["characters"] == 8


def test_http_rejects_pitch_when_streaming(client: TestClient) -> None:
    response = client.post(
        "/api/v1/services/audio/tts/SpeechSynthesizer",
        json=payload(pitch=1.2),
        headers={"X-DashScope-SSE": "enable"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "UnsupportedParameter"


def test_request_id_is_propagated(client: TestClient) -> None:
    response = client.post(
        "/api/v1/services/audio/tts/SpeechSynthesizer",
        json=payload("你好。"),
        headers={"X-Request-Id": "caller-request-id"},
    )
    assert response.status_code == 200
    assert response.json()["request_id"] == "caller-request-id"
    assert response.headers["X-Request-Id"] == "caller-request-id"
