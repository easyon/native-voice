from __future__ import annotations

import json

from fastapi.testclient import TestClient

from native_voice.api.app import create_app
from tests.fakes import FakeLLM


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _run_turn(client: TestClient, token: str) -> tuple[list[dict], int]:
    """Send one text turn and collect every JSON event plus audio bytes."""
    events: list[dict] = []
    audio_bytes = 0
    with client.websocket_connect("/ws/conversation", headers=_headers(token)) as ws:
        assert json.loads(ws.receive()["text"])["type"] == "session.ready"
        ws.send_json({"type": "session.start", "voice": "default"})
        ws.send_json({"type": "input.text", "text": "六乘以七等于多少？"})
        turn_id = None
        for _ in range(400):
            msg = ws.receive()
            if msg.get("bytes") is not None:
                audio_bytes += len(msg["bytes"])
                continue
            event = json.loads(msg["text"])
            events.append(event)
            if event["type"] == "tts.audio.start" and turn_id is None:
                turn_id = event["turn_id"]
                ws.send_json({"type": "playback.started", "turn_id": turn_id})
            if event["type"] == "metrics.turn":
                break
        if turn_id:
            ws.send_json({"type": "playback.done", "turn_id": turn_id})
    return events, audio_bytes


def test_tool_call_round_trip(cfg, token):
    llm = FakeLLM(
        script=[
            {"tool_calls": [{"name": "get_current_time", "arguments": "{}"}]},
            {"text": ["现在是", "下午三点。"]},
        ]
    )
    app = create_app(cfg, backend_kind="fake", llm=llm)
    with TestClient(app) as client:
        events, audio_bytes = _run_turn(client, token)

    kinds = [event["type"] for event in events]
    assert "assistant.tool.call" in kinds
    assert "assistant.tool.result" in kinds

    call = next(event for event in events if event["type"] == "assistant.tool.call")
    assert call["name"] == "get_current_time"
    result = next(event for event in events if event["type"] == "assistant.tool.result")
    assert result["ok"] is True
    assert "当前时间" in result["result"]

    spoken = "".join(
        event["text"] for event in events if event["type"] == "assistant.text.delta"
    )
    assert "我查一下。" in spoken  # filler before the tool ran
    assert "下午三点" in spoken
    assert audio_bytes > 0

    # The second round saw the tool result and no tools offer was needed
    # to change: tools stay available until the round limit.
    assert any(message["role"] == "tool" for message in llm.last_messages)
    tool_message = next(message for message in llm.last_messages if message["role"] == "tool")
    assert tool_message["content"] == result["result"]


def test_tool_round_limit_removes_tools(cfg, token):
    cfg.llm.max_tool_rounds = 1
    llm = FakeLLM(
        script=[
            {"tool_calls": [{"name": "get_current_time", "arguments": "{}"}]},
            {"text": ["现在是三点。"]},
        ]
    )
    app = create_app(cfg, backend_kind="fake", llm=llm)
    with TestClient(app) as client:
        events, _ = _run_turn(client, token)

    assert sum(1 for event in events if event["type"] == "assistant.tool.call") == 1
    assert llm.rounds == 2
    assert llm.last_tools is None
    spoken = "".join(
        event["text"] for event in events if event["type"] == "assistant.text.delta"
    )
    assert "现在是三点" in spoken


def test_tools_can_be_disabled(cfg, token):
    cfg.llm.tools = []
    llm = FakeLLM()
    app = create_app(cfg, backend_kind="fake", llm=llm)
    with TestClient(app) as client:
        events, audio_bytes = _run_turn(client, token)

    assert llm.last_tools is None
    assert not any(event["type"] == "assistant.tool.call" for event in events)
    assert audio_bytes > 0
