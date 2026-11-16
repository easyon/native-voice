from __future__ import annotations

from fastapi.testclient import TestClient
from funasr_runtime.main import create_app

from tests.asr_runtime.conftest import tone_pcm, wav_bytes

PATH = "/v1/audio/transcriptions"


def _post(client: TestClient, data: bytes, *, filename: str = "sample.wav", **fields):
    form = {"model": "fun-asr-realtime"}
    form.update(fields)
    return client.post(
        PATH,
        files={"file": (filename, data, "application/octet-stream")},
        data=form,
    )


def test_json_transcription(client: TestClient) -> None:
    response = _post(client, wav_bytes(tone_pcm(600)))
    assert response.status_code == 200
    body = response.json()
    assert body["text"]


def test_text_response(client: TestClient) -> None:
    response = _post(client, wav_bytes(tone_pcm(600)), response_format="text")
    assert response.status_code == 200
    assert response.text


def test_verbose_json_response(client: TestClient) -> None:
    response = _post(
        client,
        wav_bytes(tone_pcm(600)),
        response_format="verbose_json",
        language="zh",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "zh"
    assert body["duration"] > 0
    assert body["segments"]
    assert {"id", "start", "end", "text"} <= body["segments"][0].keys()


def test_srt_and_vtt(client: TestClient) -> None:
    srt = _post(client, wav_bytes(tone_pcm(600)), response_format="srt")
    assert srt.status_code == 200
    assert "-->" in srt.text
    assert "00:00:00,000" in srt.text

    vtt = _post(client, wav_bytes(tone_pcm(600)), response_format="vtt")
    assert vtt.status_code == 200
    assert vtt.text.startswith("WEBVTT")
    assert "00:00:00.000" in vtt.text


def test_pcm_upload(client: TestClient) -> None:
    response = _post(client, tone_pcm(400), filename="sample.pcm")
    assert response.status_code == 200


def test_unknown_model_uses_openai_error(client: TestClient) -> None:
    response = _post(
        client, wav_bytes(tone_pcm(200)), model="does-not-exist"
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "InvalidParameter"
    assert body["error"]["type"] == "invalid_request_error"


def test_unsupported_response_format(client: TestClient) -> None:
    response = _post(
        client, wav_bytes(tone_pcm(200)), response_format="xml"
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsupportedParameter"


def test_invalid_audio(client: TestClient) -> None:
    response = _post(client, b"not audio", filename="sample.xyz")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UnsupportedFormat"


def test_auth(config) -> None:
    config.auth.enabled = True
    config.auth.api_keys = ["test-key"]
    with TestClient(create_app(config)) as client:
        assert _post(client, wav_bytes(tone_pcm(200))).status_code == 401
        response = client.post(
            PATH,
            headers={"Authorization": "Bearer test-key"},
            files={"file": ("sample.wav", wav_bytes(tone_pcm(200)))},
            data={"model": "fun-asr-realtime"},
        )
        assert response.status_code == 200


def test_upload_limit(config) -> None:
    config.offline.max_upload_bytes = 100
    with TestClient(create_app(config)) as client:
        response = _post(client, wav_bytes(tone_pcm(200)))
        assert response.status_code == 413


def test_health_ready_and_metrics(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert "fun-asr-realtime" in ready.json()["models"]

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "funasr_ready 1" in metrics.text
    assert "funasr_requests_total" in metrics.text
