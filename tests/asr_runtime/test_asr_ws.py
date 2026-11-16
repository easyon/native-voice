from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.asr_runtime.conftest import (
    PATH,
    TASK_ID,
    finish_task,
    receive_events,
    run_task,
    sentence_events,
    tone_pcm,
    wav_bytes,
)


def _send_audio(websocket, duration_ms: int, chunk_ms: int = 100) -> None:
    pcm = tone_pcm(duration_ms)
    step = chunk_ms * 32
    for offset in range(0, len(pcm), step):
        websocket.send_bytes(pcm[offset : offset + step])


def test_full_task_partials_and_finals(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(run_task())
        assert websocket.receive_json()["header"]["event"] == "task-started"

        _send_audio(websocket, 1_200)
        websocket.send_json(finish_task())
        events = receive_events(websocket)

    names = [event["header"]["event"] for event in events]
    assert names[-1] == "task-finished"
    assert names.count("task-started") == 0
    sentences = sentence_events(events)
    finals = [sentence for sentence in sentences if sentence["sentence_end"]]
    assert finals
    assert [sentence["sentence_id"] for sentence in finals] == list(
        range(1, len(finals) + 1)
    )
    final_with_usage = [
        event["payload"]["usage"]
        for event in events
        if event["header"]["event"] == "result-generated"
        and event["payload"]["output"]["sentence"]["sentence_end"]
    ]
    assert all(usage and usage["duration"] >= 1 for usage in final_with_usage)
    assert finals[-1]["text"]

    first_of_sentence = [
        sentence for sentence in sentences if sentence["sentence_begin"]
    ]
    assert first_of_sentence[0]["sentence_end"] is False
    assert first_of_sentence[0]["end_time"] is None

    for previous, current in zip(finals, finals[1:], strict=False):
        assert current["begin_time"] >= previous["end_time"]


def test_finish_without_audio(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(run_task())
        assert websocket.receive_json()["header"]["event"] == "task-started"
        websocket.send_json(finish_task())
        events = receive_events(websocket)
    assert [event["header"]["event"] for event in events] == ["task-finished"]
    assert not sentence_events(events)


def test_finish_flushes_open_segment(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(run_task())
        websocket.receive_json()
        _send_audio(websocket, 400)
        websocket.send_json(finish_task())
        events = receive_events(websocket)
    finals = [
        sentence
        for sentence in sentence_events(events)
        if sentence["sentence_end"]
    ]
    assert len(finals) == 1
    assert finals[0]["sentence_id"] == 1
    assert finals[0]["end_time"] == 400


def test_connection_can_be_reused(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(run_task())
        assert websocket.receive_json()["header"]["event"] == "task-started"
        websocket.send_json(finish_task())
        assert receive_events(websocket)[-1]["header"]["event"] == "task-finished"

        second = run_task()
        second["header"]["task_id"] = "task-2"
        websocket.send_json(second)
        assert websocket.receive_json()["header"]["task_id"] == "task-2"
        websocket.send_json(finish_task("task-2"))
        assert receive_events(websocket)[-1]["header"]["event"] == "task-finished"


def test_unknown_model_fails_task(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(run_task(model="missing"))
        events = receive_events(websocket)
    assert events[-1]["header"]["event"] == "task-failed"
    assert events[-1]["header"]["error_code"] == "InvalidParameter"


def test_continue_task_rejected(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(run_task())
        websocket.receive_json()
        websocket.send_json(
            {
                "header": {"action": "continue-task", "task_id": TASK_ID},
                "payload": {"input": {"text": "你好"}},
            }
        )
        events = receive_events(websocket)
    assert events[-1]["header"]["event"] == "task-failed"
    assert events[-1]["header"]["error_code"] == "UnsupportedParameter"


def test_audio_before_run_task(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_bytes(tone_pcm(100))
        events = receive_events(websocket)
    assert events[-1]["header"]["event"] == "task-failed"


def test_finish_without_task(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(finish_task())
        events = receive_events(websocket)
    assert events[-1]["header"]["event"] == "task-failed"


def test_task_id_mismatch(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(run_task())
        websocket.receive_json()
        websocket.send_json(finish_task("other-task"))
        events = receive_events(websocket)
    assert events[-1]["header"]["event"] == "task-failed"


def test_websocket_auth(config) -> None:
    config.auth.enabled = True
    config.auth.api_keys = ["test-key"]
    with TestClient(config_app(config)) as client:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect(PATH):
                pass
        assert excinfo.value.code == 4401

        with client.websocket_connect(
            PATH, headers={"Authorization": "Bearer test-key"}
        ) as websocket:
            websocket.send_json(run_task())
            assert websocket.receive_json()["header"]["event"] == "task-started"
            websocket.send_json(finish_task())
            assert receive_events(websocket)[-1]["header"]["event"] == "task-finished"


def test_wav_stream_header_is_parsed(client: TestClient) -> None:
    with client.websocket_connect(PATH) as websocket:
        websocket.send_json(run_task(format="wav"))
        websocket.receive_json()
        websocket.send_bytes(wav_bytes(tone_pcm(600)))
        websocket.send_json(finish_task())
        events = receive_events(websocket)
    assert any(sentence["sentence_end"] for sentence in sentence_events(events))


def test_heartbeat(config) -> None:
    config.limits.heartbeat_interval_ms = 100
    with TestClient(config_app(config)) as client:
        with client.websocket_connect(PATH) as websocket:
            websocket.send_json(run_task(heartbeat=True))
            websocket.receive_json()
            websocket.send_bytes(tone_pcm(100))
            time.sleep(0.35)
            websocket.send_json(finish_task())
            events = receive_events(websocket)
    heartbeats = [
        sentence
        for sentence in sentence_events(events)
        if sentence.get("heartbeat")
    ]
    assert heartbeats
    assert all(sentence["sentence_id"] == 0 for sentence in heartbeats)


def test_max_segment_forces_a_split(config) -> None:
    config.engine.mock.segment_ms = 5_000
    config.limits.max_segment_ms = 1_000
    with TestClient(config_app(config)) as client:
        with client.websocket_connect(PATH) as websocket:
            websocket.send_json(run_task())
            websocket.receive_json()
            _send_audio(websocket, 1_500)
            websocket.send_json(finish_task())
            events = receive_events(websocket)
    finals = [
        sentence
        for sentence in sentence_events(events)
        if sentence["sentence_end"]
    ]
    assert [sentence["end_time"] for sentence in finals] == [1_000, 1_500]
    assert [sentence["sentence_id"] for sentence in finals] == [1, 2]


def config_app(config):
    from funasr_runtime.main import create_app

    return create_app(config)
