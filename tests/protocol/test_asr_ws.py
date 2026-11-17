from __future__ import annotations

import json

import numpy as np


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _run_task(task_id: str = "a1") -> dict:
    return {
        "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": "paraformer-local",
            "input": {},
            "parameters": {"format": "pcm", "sample_rate": 16000},
        },
    }


def _speech_pcm(seconds: float = 1.0) -> bytes:
    sr = 16000
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def _collect(ws, limit: int = 300):
    events = []
    for _ in range(limit):
        msg = ws.receive()
        text = msg.get("text")
        if text is None:
            continue
        obj = json.loads(text)
        events.append(obj)
        if obj.get("header", {}).get("event") == "task-finished":
            break
    return events


def test_asr_stream(client, token):
    with client.websocket_connect("/api-ws/v1/inference", headers=_headers(token)) as ws:
        ws.send_json(_run_task())
        assert json.loads(ws.receive()["text"])["header"]["event"] == "task-started"
        pcm = _speech_pcm(1.0)
        ws.send_bytes(pcm[: len(pcm) // 2])
        ws.send_bytes(pcm[len(pcm) // 2:])
        ws.send_json(
            {
                "header": {"action": "finish-task", "task_id": "a1", "streaming": "duplex"},
                "payload": {"input": {}},
            }
        )
        events = _collect(ws)

    results = [e for e in events if e["header"]["event"] == "result-generated"]
    assert results, "expected ASR result events"
    finals = [r for r in results if r["payload"]["output"]["sentence"]["sentence_end"]]
    assert finals and finals[-1]["payload"]["output"]["sentence"]["text"]
    finished = [e for e in events if e["header"]["event"] == "task-finished"]
    assert finished and "duration" in finished[-1]["payload"]["usage"]
