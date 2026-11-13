from __future__ import annotations

from collections.abc import Iterator

import pytest
from cosyvoice_runtime.main import create_app
from cosyvoice_runtime.settings import RuntimeConfig
from fastapi.testclient import TestClient


@pytest.fixture
def config() -> RuntimeConfig:
    config = RuntimeConfig.default()
    config.storage.type = "none"
    return config


@pytest.fixture
def client(config: RuntimeConfig) -> Iterator[TestClient]:
    with TestClient(create_app(config)) as test_client:
        yield test_client


def test_ui_is_served_and_self_contained(client: TestClient) -> None:
    response = client.get("/ui")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"

    body = response.text
    # The page calls the same contract the business uses.
    assert "/api/v1/services/audio/tts/SpeechSynthesizer" in body
    assert "/api-ws/v1/inference" in body
    assert "X-DashScope-SSE" in body
    # No CDN, fonts or other external assets: air-gapped GPU hosts must work.
    assert 'src="http' not in body
    assert 'href="http' not in body
    assert "http://" not in body
    # Core controls the smoke test exercises.
    for element_id in (
        'id="api-key"',
        'id="model"',
        'id="voice"',
        'id="text"',
        'id="mode-sync"',
        'id="mode-sse"',
        'id="mode-ws"',
        'id="synthesize"',
        'id="stop"',
        'id="log"',
    ):
        assert element_id in body


def test_ui_trailing_slash(client: TestClient) -> None:
    assert client.get("/ui/").status_code == 200


def test_ui_can_be_disabled(config: RuntimeConfig) -> None:
    config.server.ui_enabled = False
    with TestClient(create_app(config)) as test_client:
        response = test_client.get("/ui")

    assert response.status_code == 404
    assert response.json()["code"] == "NotFound"


def test_ui_stays_public_when_auth_is_enabled(config: RuntimeConfig) -> None:
    config.auth.enabled = True
    config.auth.api_keys = ["conformance-key"]
    with TestClient(create_app(config)) as test_client:
        # The static page cannot authenticate itself; the key is typed in the
        # UI and attached to API calls, so the page itself stays reachable.
        assert test_client.get("/ui").status_code == 200
        assert test_client.get("/readyz").status_code == 200
        assert test_client.post(
            "/api/v1/services/audio/tts/SpeechSynthesizer",
            json={"model": "my-tts-prod", "input": {"text": "你好"}},
        ).status_code == 401
