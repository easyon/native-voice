from __future__ import annotations

import json

import pytest
from cosyvoice_common.errors import APIError
from funasr_runtime.core.audio import TextFilter
from funasr_runtime.protocol import bailian

VALID = {
    "header": {
        "action": "run-task",
        "task_id": "2bf83b9a-baeb-4fda-8d9a-000000000001",
        "streaming": "duplex",
    },
    "payload": {
        "task_group": "audio",
        "task": "asr",
        "function": "recognition",
        "model": "fun-asr-realtime",
        "parameters": {"format": "pcm", "sample_rate": 16000},
        "input": {},
    },
}


def test_parse_run_task() -> None:
    message = bailian.parse_client_message(json.dumps(VALID))
    assert isinstance(message, bailian.RunTask)
    assert message.model == "fun-asr-realtime"
    assert message.parameters.sample_rate == 16000


def test_parse_context_text() -> None:
    payload = json.loads(json.dumps(VALID))
    payload["payload"]["input"] = {
        "context": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "阿里巴巴"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "你好"}]},
        ]
    }
    message = bailian.parse_client_message(json.dumps(payload))
    assert isinstance(message, bailian.RunTask)
    assert message.context == "阿里巴巴 你好"


def test_parse_finish_task() -> None:
    message = bailian.parse_client_message(
        json.dumps(
            {
                "header": {
                    "action": "finish-task",
                    "task_id": "t1",
                    "streaming": "duplex",
                },
                "payload": {"input": {}},
            }
        )
    )
    assert isinstance(message, bailian.FinishTask)
    assert message.task_id == "t1"


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda p: p["header"].pop("task_id"), "task_id is required"),
        (lambda p: p["payload"]["parameters"].update(format="mp3"), "unsupported"),
        (
            lambda p: p["payload"]["parameters"].update(sample_rate=8000),
            "only 16000",
        ),
        (
            lambda p: p["payload"]["parameters"].update(max_sentence_silence=100),
            "max_sentence_silence",
        ),
        (lambda p: p["payload"].update(task="tts"), "unsupported task"),
        (lambda p: p["header"].update(streaming="out"), "streaming"),
    ],
)
def test_invalid_run_task(mutate, match: str) -> None:
    payload = json.loads(json.dumps(VALID))
    mutate(payload)
    with pytest.raises(APIError, match=match):
        bailian.parse_client_message(json.dumps(payload))


def test_vocabulary_id_is_rejected() -> None:
    payload = json.loads(json.dumps(VALID))
    payload["payload"]["parameters"]["vocabulary_id"] = "hotword-list-1"
    with pytest.raises(APIError, match="vocabulary_id"):
        bailian.parse_client_message(json.dumps(payload))


def test_continue_task_rejected() -> None:
    with pytest.raises(APIError, match="continue-task"):
        bailian.parse_client_message(
            json.dumps(
                {
                    "header": {"action": "continue-task", "task_id": "t1"},
                    "payload": {},
                }
            )
        )


def test_invalid_json() -> None:
    with pytest.raises(APIError, match="invalid JSON"):
        bailian.parse_client_message("{")


def test_event_builders_match_bailian_shape() -> None:
    started = bailian.ws_task_started("t1")
    assert started == {
        "header": {"task_id": "t1", "event": "task-started", "attributes": {}},
        "payload": {},
    }

    partial = bailian.ws_sentence(
        "t1",
        sentence_id=1,
        begin_time=0,
        end_time=None,
        text="你好",
        sentence_begin=True,
        sentence_end=False,
    )
    sentence = partial["payload"]["output"]["sentence"]
    assert sentence == {
        "begin_time": 0,
        "end_time": None,
        "text": "你好",
        "heartbeat": False,
        "sentence_begin": True,
        "sentence_end": False,
        "sentence_id": 1,
        "words": [],
    }
    assert partial["payload"]["usage"] is None

    final = bailian.ws_sentence(
        "t1",
        sentence_id=1,
        begin_time=0,
        end_time=2500,
        text="你好世界。",
        sentence_begin=False,
        sentence_end=True,
        usage={"duration": 3},
    )
    assert final["payload"]["usage"] == {"duration": 3}
    assert final["payload"]["output"]["sentence"]["sentence_end"] is True

    finished = bailian.ws_task_finished("t1")
    assert finished["payload"] == {"output": {}, "usage": None}

    error = bailian.ws_task_failed(
        "t1", APIError("InvalidParameter", "bad", 400)
    )
    assert error["header"]["error_code"] == "InvalidParameter"
    assert error["header"]["error_message"] == "bad"


def test_special_word_filter() -> None:
    text_filter = TextFilter.parse(
        json.dumps(
            {
                "filter_with_signed": {"word_list": ["项目"]},
                "filter_with_empty": {"word_list": ["报告"]},
            }
        )
    )
    assert text_filter.apply("项目报告完成") == "**完成"


def test_special_word_filter_limits() -> None:
    words = [f"w{index}" for index in range(33)]
    with pytest.raises(APIError, match="at most 32"):
        TextFilter.parse(
            json.dumps({"filter_with_signed": {"word_list": words}})
        )
