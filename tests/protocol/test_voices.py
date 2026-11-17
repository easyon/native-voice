from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from native_voice.security.keystore import KeyStore


def _wav_bytes(seconds: float = 3.0, sr: int = 16000) -> io.BytesIO:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_clone_with_text(client, token, cfg):
    resp = client.post(
        "/v1/voices/clone",
        headers=_headers(token),
        files={"file": ("ref.wav", _wav_bytes(), "audio/wav")},
        data={"name": "测试音色", "text": "你好，世界。", "consent": "true"},
    )
    assert resp.status_code == 200, resp.text
    obj = resp.json()
    voice_id = obj["voice"]["name"]
    assert voice_id.startswith("clone_")
    assert obj["transcript"] == "你好，世界。"
    assert (cfg.voice_store.dir / f"{voice_id}.wav").exists()
    voices = client.get("/v1/voices").json()["voices"]
    assert any(v["name"] == voice_id for v in voices)

    resp = client.delete(f"/v1/voices/{voice_id}", headers=_headers(token))
    assert resp.status_code == 200
    assert not (cfg.voice_store.dir / f"{voice_id}.wav").exists()
    voices = client.get("/v1/voices").json()["voices"]
    assert not any(v["name"] == voice_id for v in voices)


def test_clone_auto_transcript(client, token):
    resp = client.post(
        "/v1/voices/clone",
        headers=_headers(token),
        files={"file": ("ref.wav", _wav_bytes(3.0), "audio/wav")},
        data={"name": "自动识别", "text": "", "consent": "true"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["transcript"]


def test_clone_requires_consent(client, token):
    resp = client.post(
        "/v1/voices/clone",
        headers=_headers(token),
        files={"file": ("ref.wav", _wav_bytes(), "audio/wav")},
        data={"name": "x", "text": "你好", "consent": "false"},
    )
    assert resp.status_code == 400


def test_clone_requires_tts_scope(client, cfg):
    store = KeyStore(cfg.auth.db_path)
    _, token = store.create_key("asr-only", ["asr"])
    store.close()
    resp = client.post(
        "/v1/voices/clone",
        headers=_headers(token),
        files={"file": ("ref.wav", _wav_bytes(), "audio/wav")},
        data={"name": "x", "text": "你好", "consent": "true"},
    )
    assert resp.status_code == 403
