from __future__ import annotations

import json

import pytest
from starlette.websockets import WebSocketDisconnect

from native_voice.security.keystore import KeyStore


def test_missing_key_rejected(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api-ws/v1/inference"):
            pass


def test_bad_key_rejected(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/api-ws/v1/inference", headers={"Authorization": "Bearer nv_deadbeef_nope"}
        ):
            pass


def test_scope_rejected(client, cfg):
    store = KeyStore(cfg.auth.db_path)
    _, token = store.create_key("asr-only", ["asr"])
    store.close()
    with client.websocket_connect(
        "/api-ws/v1/inference", headers={"Authorization": f"Bearer {token}"}
    ) as ws:
        ws.send_json(
            {
                "header": {"action": "run-task", "task_id": "s1", "streaming": "duplex"},
                "payload": {
                    "task_group": "audio",
                    "task": "tts",
                    "function": "SpeechSynthesizer",
                    "model": "x",
                    "input": {},
                    "parameters": {"text_type": "PlainText", "voice": "default"},
                },
            }
        )
        msg = json.loads(ws.receive()["text"])
        assert msg["header"]["event"] == "task-failed"
        assert msg["header"]["error_code"] == "Forbidden"


def test_oversized_audio_frame(client, token):
    with client.websocket_connect(
        "/api-ws/v1/inference", headers={"Authorization": f"Bearer {token}"}
    ) as ws:
        ws.send_json(
            {
                "header": {"action": "run-task", "task_id": "s2", "streaming": "duplex"},
                "payload": {
                    "task_group": "audio",
                    "task": "asr",
                    "function": "recognition",
                    "model": "x",
                    "input": {},
                    "parameters": {"format": "pcm", "sample_rate": 16000},
                },
            }
        )
        assert json.loads(ws.receive()["text"])["header"]["event"] == "task-started"
        ws.send_bytes(b"\x00" * 70000)
        msg = json.loads(ws.receive()["text"])
        assert msg["header"]["event"] == "task-failed"
        assert msg["header"]["error_code"] == "InvalidParameter"
