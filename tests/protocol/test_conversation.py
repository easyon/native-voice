from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from native_voice.api.app import create_app
from tests.fakes import FakeLLM


def _pcm_utterance() -> bytes:
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    t = np.arange(sr, dtype=np.float32) / sr
    speech = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    tail = np.zeros(sr, dtype=np.float32)
    audio = np.concatenate([silence, speech, tail])
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_conversation_bearer(cfg, token):
    app = create_app(cfg, backend_kind="fake", llm=FakeLLM())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/conversation", headers=_headers(token)) as ws:
            ready = json.loads(ws.receive()["text"])
            assert ready["type"] == "session.ready"
            ws.send_json({"type": "session.start", "voice": "default"})
            pcm = _pcm_utterance()
            step = 640
            for i in range(0, len(pcm), step):
                ws.send_bytes(pcm[i:i + step])

            seen = []
            turn_id = None
            audio_bytes = 0
            for _ in range(300):
                msg = ws.receive()
                if msg.get("bytes") is not None:
                    audio_bytes += len(msg["bytes"])
                    continue
                obj = json.loads(msg["text"])
                seen.append(obj["type"])
                if obj["type"] == "tts.audio.start" and turn_id is None:
                    turn_id = obj["turn_id"]
                    ws.send_json({"type": "playback.started", "turn_id": turn_id})
                if obj["type"] == "metrics.turn":
                    break
            if turn_id:
                ws.send_json({"type": "playback.done", "turn_id": turn_id})

    assert "asr.final" in seen
    assert "assistant.text.delta" in seen
    assert audio_bytes > 0
    assert "metrics.turn" in seen


def test_conversation_ticket(cfg, token):
    app = create_app(cfg, backend_kind="fake", llm=FakeLLM())
    with TestClient(app) as client:
        resp = client.post("/v1/realtime/ticket", headers=_headers(token))
        assert resp.status_code == 200
        ticket = resp.json()["ticket"]
        with client.websocket_connect("/ws/conversation") as ws:
            ws.send_json({"type": "auth", "ticket": ticket})
            ready = json.loads(ws.receive()["text"])
            assert ready["type"] == "session.ready"
            ws.send_json({"type": "session.end"})
