from __future__ import annotations

import io
import json
import math
import wave
from array import array
from collections.abc import Iterator
from pathlib import Path

import pytest
from cosyvoice_runtime.main import create_app
from cosyvoice_runtime.settings import RuntimeConfig
from fastapi.testclient import TestClient


@pytest.fixture
def config(tmp_path: Path) -> RuntimeConfig:
    config = RuntimeConfig.default()
    config.storage.type = "none"
    config.cloning.dir = str(tmp_path / "voices")
    config.cloning.min_duration_s = 0.5
    return config


@pytest.fixture
def client(config: RuntimeConfig) -> Iterator[TestClient]:
    with TestClient(create_app(config)) as test_client:
        yield test_client


def wav_bytes(seconds: float = 2.0, rate: int = 24_000) -> bytes:
    frames = int(seconds * rate)
    samples = array(
        "h",
        (int(8_000 * math.sin(2 * math.pi * 440 * index / rate)) for index in range(frames)),
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(samples.tobytes())
    return buffer.getvalue()


def clone(client: TestClient, **data: str):
    files = {"audio": ("reference.wav", wav_bytes(), "audio/wav")}
    form = {
        "transcript": "你好，这是一段参考音频。",
        "name": "测试音色",
        "source": "upload",
    }
    form.update(data)
    return client.post("/api/v1/voices/clone", files=files, data=form)


def synthesize(client: TestClient, voice: str, text: str = "你好"):
    return client.post(
        "/api/v1/services/audio/tts/SpeechSynthesizer",
        json={
            "model": "my-tts-prod",
            "input": {"text": text, "voice": voice},
        },
    )


def test_list_voices_includes_configured_voice(client: TestClient) -> None:
    response = client.get("/api/v1/voices")

    assert response.status_code == 200
    voices = response.json()["voices"]
    assert any(voice["id"] == "voice-a" and not voice["cloned"] for voice in voices)


def test_clone_voice_registers_and_synthesizes(
    client: TestClient, config: RuntimeConfig
) -> None:
    response = clone(client)

    assert response.status_code == 200, response.text
    voice = response.json()["voice"]
    assert voice["id"].startswith("clone-voice-")
    assert voice["cloned"] is True
    assert voice["duration_s"] == pytest.approx(2.0, abs=0.05)

    listed = client.get("/api/v1/voices").json()["voices"]
    assert any(item["id"] == voice["id"] and item["cloned"] for item in listed)

    synthesis = synthesize(client, voice["id"])
    assert synthesis.status_code == 200, synthesis.text

    directory = Path(config.cloning.dir) / voice["id"]
    assert (directory / "prompt.wav").is_file()
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    assert meta["prompt_text"].startswith("<|endofprompt|>")
    assert meta["transcript"] == "你好，这是一段参考音频。"
    assert meta["source"] == "upload"


def test_cloned_voice_survives_restart(config: RuntimeConfig) -> None:
    with TestClient(create_app(config)) as first:
        voice_id = clone(first).json()["voice"]["id"]

    restarted = config.model_copy(deep=True)
    with TestClient(create_app(restarted)) as second:
        listed = second.get("/api/v1/voices").json()["voices"]
        assert any(item["id"] == voice_id and item["cloned"] for item in listed)
        assert synthesize(second, voice_id).status_code == 200


def test_clone_requires_auth_when_enabled(config: RuntimeConfig) -> None:
    config.auth.enabled = True
    config.auth.api_keys = ["test-key"]
    with TestClient(create_app(config)) as test_client:
        assert clone(test_client).status_code == 401
        assert test_client.get("/api/v1/voices").status_code == 401

        headers = {"Authorization": "Bearer test-key"}
        files = {"audio": ("reference.wav", wav_bytes(), "audio/wav")}
        authorized = test_client.post(
            "/api/v1/voices/clone",
            files=files,
            data={"transcript": "你好。"},
            headers=headers,
        )
        assert authorized.status_code == 200, authorized.text
        assert test_client.get("/api/v1/voices", headers=headers).status_code == 200


def test_clone_rejects_short_audio(client: TestClient, config: RuntimeConfig) -> None:
    files = {"audio": ("short.wav", wav_bytes(0.2), "audio/wav")}
    response = client.post(
        "/api/v1/voices/clone",
        files=files,
        data={"transcript": "你好。"},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "InvalidParameter"
    # Nothing should be left behind after a rejected upload.
    assert not list(Path(config.cloning.dir).glob("clone-*"))


def test_clone_rejects_non_audio(client: TestClient) -> None:
    files = {"audio": ("notes.txt", b"not audio", "text/plain")}
    response = client.post(
        "/api/v1/voices/clone",
        files=files,
        data={"transcript": "你好。"},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "UnsupportedFormat"


def test_clone_rejects_oversized_upload(client: TestClient, config: RuntimeConfig) -> None:
    config.cloning.max_upload_bytes = 2048
    files = {"audio": ("reference.wav", wav_bytes(), "audio/wav")}
    response = client.post(
        "/api/v1/voices/clone",
        files=files,
        data={"transcript": "你好。"},
    )

    assert response.status_code == 413
    assert response.json()["code"] == "RequestTooLarge"


def test_delete_cloned_voice(client: TestClient) -> None:
    voice_id = clone(client).json()["voice"]["id"]

    deleted = client.delete(f"/api/v1/voices/{voice_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    listed = client.get("/api/v1/voices").json()["voices"]
    assert all(item["id"] != voice_id for item in listed)
    synthesis = synthesize(client, voice_id)
    assert synthesis.status_code == 400
    assert synthesis.json()["code"] == "VoiceNotFound"


def test_delete_configured_voice_is_rejected(client: TestClient) -> None:
    response = client.delete("/api/v1/voices/voice-a")

    assert response.status_code == 404
    assert response.json()["code"] == "NotFound"


def test_cloning_can_be_disabled(config: RuntimeConfig) -> None:
    config.cloning.enabled = False
    with TestClient(create_app(config)) as test_client:
        response = clone(test_client)

    assert response.status_code == 403
    assert response.json()["code"] == "AccessDenied"
