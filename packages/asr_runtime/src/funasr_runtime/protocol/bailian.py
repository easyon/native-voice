from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from cosyvoice_common.errors import (
    APIError,
    invalid_parameter,
    unsupported_parameter,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.audio import validate_stream_format


class AsrParameters(BaseModel):
    model_config = ConfigDict(extra="ignore")

    format: str = "pcm"
    sample_rate: int = 16_000
    max_sentence_silence: int = 1_300
    heartbeat: bool = False
    language_hints: list[str] = Field(default_factory=list)
    speech_noise_threshold: float = 0.0
    special_word_filter: str | None = None
    hotwords: str | None = None
    itn: bool | None = None
    vocabulary_id: str | None = None
    multi_threshold_mode_enabled: bool = False


@dataclass(frozen=True, slots=True)
class RunTask:
    task_id: str
    model: str
    parameters: AsrParameters
    input: dict[str, Any]
    context: str = ""


@dataclass(frozen=True, slots=True)
class FinishTask:
    task_id: str


ClientMessage = RunTask | FinishTask


def parse_client_message(raw: str) -> ClientMessage:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise invalid_parameter(f"invalid JSON message: {exc.msg}") from exc
    if not isinstance(message, dict):
        raise invalid_parameter("message must be a JSON object")

    header = message.get("header")
    if not isinstance(header, dict):
        raise invalid_parameter("header must be a JSON object")
    action = header.get("action")
    task_id = _task_id(header)
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise invalid_parameter("payload must be a JSON object")

    if action == "run-task":
        streaming = header.get("streaming")
        if streaming not in (None, "duplex"):
            raise unsupported_parameter("only streaming=duplex is supported")
        for key, expected in (
            ("task_group", "audio"),
            ("task", "asr"),
            ("function", "recognition"),
        ):
            value = payload.get(key)
            if value != expected:
                raise invalid_parameter(f"unsupported {key}: {value!r}")
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise invalid_parameter("payload.model is required")
        raw_parameters = payload.get("parameters") or {}
        if not isinstance(raw_parameters, dict):
            raise invalid_parameter("payload.parameters must be a JSON object")
        try:
            parameters = AsrParameters.model_validate(raw_parameters)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            location = ".".join(str(part) for part in first.get("loc", ()))
            detail = first.get("msg", "invalid parameters")
            raise invalid_parameter(
                f"invalid parameters.{location}: {detail}" if location else detail
            ) from exc

        validate_stream_format(parameters.format)
        if parameters.sample_rate != 16_000:
            raise unsupported_parameter(
                f"unsupported sample_rate {parameters.sample_rate}; only 16000 is supported"
            )
        if not 200 <= parameters.max_sentence_silence <= 6_000:
            raise invalid_parameter(
                "parameters.max_sentence_silence must be between 200 and 6000 ms"
            )
        if not -1.0 <= parameters.speech_noise_threshold <= 1.0:
            raise invalid_parameter(
                "parameters.speech_noise_threshold must be between -1.0 and 1.0"
            )
        if parameters.vocabulary_id:
            raise unsupported_parameter(
                "parameters.vocabulary_id is not supported; "
                "pass hotwords directly or configure model.hotwords"
            )
        input_ = payload.get("input") or {}
        if not isinstance(input_, dict):
            raise invalid_parameter("payload.input must be a JSON object")
        return RunTask(
            task_id=task_id,
            model=model.strip(),
            parameters=parameters,
            input=input_,
            context=context_text(input_),
        )

    if action == "finish-task":
        streaming = header.get("streaming")
        if streaming not in (None, "duplex"):
            raise unsupported_parameter("only streaming=duplex is supported")
        input_ = payload.get("input") or {}
        if not isinstance(input_, dict):
            raise invalid_parameter("payload.input must be a JSON object")
        return FinishTask(task_id=task_id)

    if action == "continue-task":
        raise unsupported_parameter(
            "continue-task is not used for ASR; send binary audio frames"
        )
    raise invalid_parameter(
        "unsupported action; expected run-task or finish-task"
    )


def _task_id(header: dict[str, Any]) -> str:
    task_id = str(header.get("task_id") or "").strip()
    if not task_id:
        raise invalid_parameter("header.task_id is required")
    if len(task_id) > 128 or any(not ch.isprintable() for ch in task_id):
        raise invalid_parameter("header.task_id is invalid or too long")
    return task_id


def context_text(input_: dict[str, Any]) -> str:
    """Extract best-effort text from the Bailian context array for hotwords."""
    context = input_.get("context")
    if not isinstance(context, list):
        return ""
    parts: list[str] = []
    for item in context:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list):
            for entry in content:
                if isinstance(entry, dict):
                    text = entry.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        elif isinstance(content, str):
            parts.append(content)
    return " ".join(part.strip() for part in parts if part.strip())[:2_000]


def ws_task_started(task_id: str) -> dict[str, Any]:
    return {
        "header": {"task_id": task_id, "event": "task-started", "attributes": {}},
        "payload": {},
    }


def ws_sentence(
    task_id: str,
    *,
    sentence_id: int,
    begin_time: int,
    end_time: int | None,
    text: str,
    sentence_begin: bool,
    sentence_end: bool,
    heartbeat: bool = False,
    words: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "header": {"task_id": task_id, "event": "result-generated", "attributes": {}},
        "payload": {
            "output": {
                "sentence": {
                    "begin_time": int(begin_time),
                    "end_time": None if end_time is None else int(end_time),
                    "text": text,
                    "heartbeat": bool(heartbeat),
                    "sentence_begin": bool(sentence_begin),
                    "sentence_end": bool(sentence_end),
                    "sentence_id": int(sentence_id),
                    "words": words or [],
                }
            },
            "usage": usage,
        },
    }


def ws_task_finished(task_id: str) -> dict[str, Any]:
    return {
        "header": {"task_id": task_id, "event": "task-finished", "attributes": {}},
        "payload": {"output": {}, "usage": None},
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


def words_payload(words: Any) -> list[dict[str, Any]]:
    result = []
    for word in words or ():
        result.append(
            {
                "begin_time": int(word.begin_time),
                "end_time": int(word.end_time),
                "text": word.text,
                "punctuation": word.punctuation,
            }
        )
    return result
