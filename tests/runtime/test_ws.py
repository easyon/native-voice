from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from cosyvoice_runtime.main import create_app
from cosyvoice_runtime.settings import RuntimeConfig
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

PATH = "/api-ws/v1/inference"
TASK_ID = "2bf83b9a-baeb-4fda-8d9a-000000000001"


@pytest.fixture
def config() -> RuntimeConfig:
    config = RuntimeConfig.default()
    config.storage.type = "none"
    return config


@pytest.fixture
def client(config: RuntimeConfig) -> Iterator[TestClient]:
    with TestClient(create_app(config)) as test_client:
        yield test_client


def run_task(model: str = "my-tts-prod") -> dict:
    return {
        "header": {
            "action": "run-task",
            "task_id": TASK_ID,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "model": model,
            "parameters": {
                "voice": "voice-a",
                "format": "pcm",
                "sample_rate": 24000,
            },
            "input": {},
        },
    }


def continue_task(text: str) -> dict:
    return {
        "header": {
            "action": "continue-task",
            "task_id": TASK_ID,
            "streaming": "duplex",
        },
        "payload": {"input": {"text": text}},
    }


def finish_task(cancel: bool = False) -> dict:
    return {
        "header": {
            "action": "finish-task",
            "task_id": TASK_ID,
            "streaming": "duplex",
        },
        "payload": {"input": {"directive": "cancel"}} if cancel else {"input": {}},
    }


def test_ws_full_task_and_binary_audio(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(run_task())
        assert websocket.receive_json()["header"]["event"] == "task-started"

        websocket.send_json(continue_task("你好。今天好吗？"))
        websocket.send_json(finish_task())

        types: list[str] = []
        binary_frames = 0
        finished = False
        while not finished:
            message = websocket.receive()
            text = message.get("text")
            data = message.get("bytes")
            if data is not None:
                binary_frames += 1
                assert data
                continue
            if text is None:
                continue
            event = json.loads(text)
            name = event["header"]["event"]
            if name == "result-generated":
                types.append(event["payload"]["output"]["type"])
            elif name == "task-finished":
                finished = True
            elif name == "task-failed":
                pytest.fail(event["header"].get("error_message"))

        assert types.count("sentence-begin") == 2
        assert types.count("sentence-end") == 2
        assert types.count("sentence-synthesis") == binary_frames > 0
        assert types[0] == "sentence-begin"
        assert types[-1] == "sentence-end"


def test_ws_unknown_model_fails_task(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(run_task(model="missing"))
        event = websocket.receive_json()
        assert event["header"]["event"] == "task-failed"
        assert event["header"]["error_code"] == "InvalidParameter"


def test_ws_task_id_mismatch_fails(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(run_task())
        websocket.receive_json()
        message = continue_task("你好。")
        message["header"]["task_id"] = "another-task"
        websocket.send_json(message)
        event = websocket.receive_json()
        while event["header"]["event"] == "result-generated":
            event = websocket.receive_json()
        assert event["header"]["event"] == "task-failed"
        assert event["header"]["error_code"] == "InvalidParameter"


def test_ws_auth(client: TestClient) -> None:
    config = RuntimeConfig.default()
    config.storage.type = "none"
    config.auth.enabled = True
    config.auth.api_keys = ["test-key"]
    with TestClient(create_app(config)) as auth_client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with auth_client.websocket_connect(PATH):
                pass
        assert exc.value.code == 4401

        with auth_client.websocket_connect(
            PATH,
            headers={"Authorization": "Bearer test-key"},
        ) as websocket:
            websocket.send_json(run_task())
            assert websocket.receive_json()["header"]["event"] == "task-started"


def test_ws_cancel(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(run_task())
        websocket.receive_json()
        websocket.send_json(continue_task("你好。"))
        websocket.send_json(finish_task(cancel=True))
        while True:
            message = websocket.receive()
            if message.get("bytes") is not None:
                continue
            text = message.get("text")
            if text is None:
                continue
            event = json.loads(text)
            name = event["header"]["event"]
            if name == "task-finished":
                break
            assert name != "task-failed"


def test_ws_cancel_while_audio_is_streaming() -> None:
    """Cancellation must stop the runner even while a send is in flight.

    Regression test: asyncio.wait_for can swallow CancelledError when the
    inner operation completes at the same time, which used to leave the
    runner alive and make finish-task hang forever.
    """
    config = RuntimeConfig.default()
    config.storage.type = "none"
    config.engine.mock.realtime = True
    config.engine.mock.chunk_ms = 5
    with TestClient(create_app(config)) as streaming_client:
        with streaming_client.websocket_connect(PATH) as websocket:
            websocket.send_json(run_task())
            websocket.receive_json()
            websocket.send_json(continue_task("你好。今天天气怎么样？"))
            while True:
                message = websocket.receive()
                if message.get("bytes") is not None:
                    break
            websocket.send_json(finish_task(cancel=True))
            while True:
                message = websocket.receive()
                text = message.get("text")
                if text is None:
                    continue
                event = json.loads(text)
                name = event["header"]["event"]
                if name == "task-finished":
                    break
                assert name != "task-failed"
