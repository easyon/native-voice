from __future__ import annotations

import json


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _run_task(task_id: str = "t1") -> dict:
    return {
        "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
        "payload": {
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "model": "cosyvoice-local",
            "input": {},
            "parameters": {
                "text_type": "PlainText",
                "voice": "default",
                "format": "pcm",
                "sample_rate": 24000,
            },
        },
    }


def _collect(ws, limit: int = 200):
    events = []
    for _ in range(limit):
        msg = ws.receive()
        if msg.get("bytes") is not None:
            events.append(("bytes", msg["bytes"]))
            continue
        text = msg.get("text")
        if text is None:
            continue
        obj = json.loads(text)
        events.append(("json", obj))
        if obj.get("header", {}).get("event") == "task-finished":
            break
    return events


def test_tts_stream(client, token):
    with client.websocket_connect("/api-ws/v1/inference", headers=_headers(token)) as ws:
        ws.send_json(_run_task())
        first = json.loads(ws.receive()["text"])
        assert first["header"]["event"] == "task-started"

        ws.send_json(
            {
                "header": {"action": "continue-task", "task_id": "t1", "streaming": "duplex"},
                "payload": {"input": {"text": "你好。"}},
            }
        )
        ws.send_json(
            {
                "header": {"action": "finish-task", "task_id": "t1", "streaming": "duplex"},
                "payload": {"input": {}},
            }
        )
        events = _collect(ws)

    kinds = [
        e[1]["payload"]["output"]["type"]
        for e in events
        if e[0] == "json" and e[1]["header"]["event"] == "result-generated"
    ]
    assert "sentence-begin" in kinds
    assert "sentence-synthesis" in kinds
    assert "sentence-end" in kinds
    audio = b"".join(e[1] for e in events if e[0] == "bytes")
    assert len(audio) > 0 and len(audio) % 2 == 0
    finished = [e[1] for e in events if e[0] == "json" and e[1]["header"]["event"] == "task-finished"][-1]
    assert finished["payload"]["usage"]["characters"] == 3


def test_tts_unsupported_ssml(client, token):
    with client.websocket_connect("/api-ws/v1/inference", headers=_headers(token)) as ws:
        payload = _run_task("t2")
        payload["payload"]["parameters"]["enable_ssml"] = True
        ws.send_json(payload)
        msg = json.loads(ws.receive()["text"])
        assert msg["header"]["event"] == "task-failed"
        assert msg["header"]["error_code"] == "UnsupportedFeature"
