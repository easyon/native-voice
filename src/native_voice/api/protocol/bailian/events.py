"""Bailian-compatible WebSocket event schemas and response builders."""
from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .errors import ProtocolError


class EventHeader(BaseModel):
    model_config = ConfigDict(extra="allow")
    action: str
    task_id: str = ""
    streaming: str = "duplex"


class ClientEvent(BaseModel):
    model_config = ConfigDict(extra="allow")
    header: EventHeader
    payload: dict[str, Any] = Field(default_factory=dict)


def parse_client_event(text: str) -> ClientEvent:
    try:
        raw = json.loads(text)
        return ClientEvent.model_validate(raw)
    except Exception as exc:
        raise ProtocolError(f"invalid event: {exc}") from exc


# ----------------------------------------------------------------- ASR server
def task_started(task_id: str) -> dict:
    return {"header": {"task_id": task_id, "event": "task-started", "attributes": {}}, "payload": {}}


def asr_result(task_id: str, sentence: dict, usage: dict | None = None) -> dict:
    return {
        "header": {"task_id": task_id, "event": "result-generated", "attributes": {}},
        "payload": {"output": {"sentence": sentence}, "usage": usage},
    }


def task_finished(task_id: str, usage: dict | None = None) -> dict:
    return {
        "header": {
            "task_id": task_id,
            "event": "task-finished",
            "attributes": {"request_uuid": str(uuid.uuid4())},
        },
        "payload": {"usage": usage or {}},
    }


def task_failed(task_id: str, code: str, message: str) -> dict:
    return {
        "header": {
            "task_id": task_id,
            "event": "task-failed",
            "error_code": code,
            "error_message": message,
            "attributes": {},
        },
        "payload": {},
    }


# ----------------------------------------------------------------- TTS server
def tts_result(
    task_id: str,
    kind: str,
    index: int,
    original_text: str,
    words: list | None = None,
    usage: dict | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "output": {
            "type": kind,
            "sentence": {"index": index, "words": words or []},
            "original_text": original_text,
        }
    }
    if usage is not None:
        payload["usage"] = usage
    return {"header": {"task_id": task_id, "event": "result-generated", "attributes": {}}, "payload": payload}
