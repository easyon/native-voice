from __future__ import annotations

from fastapi.testclient import TestClient
from funasr_runtime.main import create_app


def test_ui_is_served(client: TestClient) -> None:
    response = client.get("/ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert "FunASR 测试台" in response.text
    assert "api-ws/v1/inference" in response.text
    assert "v1/audio/transcriptions" in response.text
    assert "sentence_begin id=" in response.text
    assert "sentence_end id=" in response.text


def test_ui_can_be_disabled(config) -> None:
    config.server.ui_enabled = False
    with TestClient(create_app(config)) as client:
        assert client.get("/ui").status_code == 404
