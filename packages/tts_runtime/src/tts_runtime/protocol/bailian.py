from __future__ import annotations

import json
from typing import Any

from cosyvoice_common.errors import APIError

from ..storage import StoredAudio


def sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ) + "\n\n"


def http_success(
    request_id: str,
    *,
    audio_b64: str,
    stored: StoredAudio | None,
    characters: int,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "output": {
            "finish_reason": "stop",
            "audio": {
                "data": audio_b64,
                "url": stored.url if stored else "",
                "id": stored.id if stored else "",
                "expires_at": stored.expires_at if stored else 0,
            },
        },
        "usage": {"characters": characters},
    }


def sentence_begin(
    request_id: str,
    index: int,
    text: str,
    words: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "output": {
            "finish_reason": None,
            "type": "sentence-begin",
            "original_text": text,
            "sentence": {"index": index, "words": words},
            "audio": {"data": ""},
        },
        "usage": {"characters": 0},
    }


def sentence_synthesis(
    request_id: str,
    index: int,
    words: list[dict[str, Any]],
    audio_b64: str,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "output": {
            "finish_reason": None,
            "type": "sentence-synthesis",
            "sentence": {"index": index, "words": words},
            "audio": {"data": audio_b64},
        },
        "usage": {"characters": 0},
    }


def sentence_end(
    request_id: str,
    index: int,
    text: str,
    words: list[dict[str, Any]],
    characters: int,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "output": {
            "finish_reason": None,
            "type": "sentence-end",
            "original_text": text,
            "sentence": {"index": index, "words": words},
            "audio": {"data": ""},
        },
        "usage": {"characters": characters},
    }


def http_stream_finish(
    request_id: str,
    *,
    stored: StoredAudio | None,
    characters: int,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "output": {
            "finish_reason": "stop",
            "audio": {
                "data": "",
                "url": stored.url if stored else "",
                "id": stored.id if stored else "",
                "expires_at": stored.expires_at if stored else 0,
            },
        },
        "usage": {"characters": characters},
    }


def error_body(request_id: str, error: APIError) -> dict[str, Any]:
    return error.to_dict(request_id)


def ws_task_started(task_id: str) -> dict[str, Any]:
    return {
        "header": {
            "task_id": task_id,
            "event": "task-started",
            "attributes": {},
        },
        "payload": {},
    }


def ws_result(
    task_id: str,
    output: dict[str, Any],
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"output": output}
    if usage is not None:
        payload["usage"] = usage
    return {
        "header": {
            "task_id": task_id,
            "event": "result-generated",
            "attributes": {},
        },
        "payload": payload,
    }


def ws_task_finished(
    task_id: str,
    request_uuid: str,
    characters: int | None = None,
) -> dict[str, Any]:
    # The legacy v1 SDK (dashscope.audio.tts.SpeechSynthesizer) treats any
    # non-empty task-finished payload as a sentence event and raises KeyError
    # on it, so one-shot "streaming=out" tasks finish with an empty payload.
    payload: dict[str, Any] = (
        {} if characters is None else {"usage": {"characters": characters}}
    )
    return {
        "header": {
            "task_id": task_id,
            "event": "task-finished",
            "attributes": {"request_uuid": request_uuid},
        },
        "payload": payload,
    }


def ws_task_failed(task_id: str, error: APIError) -> dict[str, Any]:
    return {
        "header": {
            "task_id": task_id,
            "event": "task-failed",
            "error_code": error.code,
            "error_message": error.message,
            "attributes": {},
        },
        "payload": {},
    }
