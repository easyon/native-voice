from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from cosyvoice_common.errors import (
    APIError,
    invalid_parameter,
    normalize_error,
)
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from ..context import bind_context
from ..core.session import WSSession
from ..core.tenant import Tenant
from ..protocol import bailian
from ..schemas import SpeechInput
from .deps import authenticate, runtime_from_websocket

logger = logging.getLogger("tts_runtime.ws")
router = APIRouter()

_INTERNAL_TASK_ID = "00000000-0000-0000-0000-000000000000"


class WebSocketSender:
    """Serializes JSON and binary frames on one connection.

    ``audio`` writes the metadata and its binary frame under a single lock so a
    concurrent error frame cannot land between them.
    """

    def __init__(self, websocket: WebSocket, send_timeout_s: float = 10.0) -> None:
        self._websocket = websocket
        self._lock = asyncio.Lock()
        self._timeout = max(0.1, float(send_timeout_s))

    async def json(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            await asyncio.wait_for(
                self._websocket.send_json(payload),
                timeout=self._timeout,
            )

    async def bytes(self, data: bytes) -> None:
        async with self._lock:
            await asyncio.wait_for(
                self._websocket.send_bytes(data),
                timeout=self._timeout,
            )

    async def audio(self, payload: dict[str, Any], data: bytes) -> None:
        async with self._lock:
            await asyncio.wait_for(
                self._websocket.send_json(payload),
                timeout=self._timeout,
            )
            await asyncio.wait_for(
                self._websocket.send_bytes(data),
                timeout=self._timeout,
            )


@router.websocket("/api-ws/v1/inference")
async def inference(websocket: WebSocket) -> None:
    runtime = runtime_from_websocket(websocket)
    try:
        tenant = authenticate(
            websocket.headers.get("authorization"),
            runtime.config.auth,
        )
    except APIError as exc:
        logger.info("websocket auth failed: %s", exc.code)
        await websocket.close(code=4401)
        return

    try:
        await runtime.websocket_connected()
    except APIError:
        await websocket.close(code=1013)
        return

    try:
        await websocket.accept()
    except Exception:
        await runtime.websocket_disconnected()
        raise
    sender = WebSocketSender(
        websocket,
        send_timeout_s=runtime.config.server.ws_send_timeout_s,
    )
    session = WSSession(
        orchestrator=runtime.orchestrator,
        sender=sender,
        limits=runtime.config.limits,
        metrics=runtime.metrics,
        lifecycle=runtime.lifecycle,
        limiter=runtime.limiter,
    )
    idle_timeout = max(1.0, float(runtime.config.server.ws_idle_timeout_s))
    max_message = max(1, int(runtime.config.server.max_ws_message_bytes))

    logger.info("websocket connected: tenant=%s", tenant.id)
    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=idle_timeout,
                )
            except asyncio.TimeoutError:
                # A long synthesis may legitimately produce no inbound frames
                # for more than the idle window. Keep the connection while a
                # task is active; its own deadline still bounds the task.
                if session.task is not None:
                    continue
                logger.info("websocket idle timeout: tenant=%s", tenant.id)
                break

            if message["type"] == "websocket.disconnect":
                break
            raw = message.get("text")
            if raw is None:
                logger.warning("ignoring binary message from client")
                continue
            if len(raw.encode("utf-8")) > max_message:
                await _report_error(
                    sender,
                    session,
                    _task_id_of(raw),
                    invalid_parameter("websocket message is too large"),
                )
                continue

            task_id = _task_id_of(raw)
            try:
                with bind_context(
                    request_id=task_id,
                    tenant=tenant.id,
                    task_id=task_id,
                    api="ws",
                ):
                    await _dispatch(raw, sender, session, runtime, tenant)
            except APIError as exc:
                await _report_error(sender, session, _task_id_of(raw), exc)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - keep a bad connection from killing the worker
        logger.exception("websocket connection failed")
    finally:
        await session.close()
        await runtime.websocket_disconnected()
        logger.info("websocket disconnected: tenant=%s", tenant.id)


async def _dispatch(
    raw: str,
    sender: WebSocketSender,
    session: WSSession,
    runtime: Any,
    tenant: Tenant,
) -> None:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise invalid_parameter(f"invalid JSON message: {exc.msg}") from exc
    if not isinstance(message, dict):
        raise invalid_parameter("message must be a JSON object")

    header = message.get("header") or {}
    if not isinstance(header, dict):
        raise invalid_parameter("header must be a JSON object")
    action = header.get("action")
    task_id = str(header.get("task_id") or "").strip()
    if len(task_id) > 128 or any(not ch.isprintable() for ch in task_id):
        raise invalid_parameter("header.task_id is invalid or too long")
    if action != "run-task" and not task_id:
        raise invalid_parameter("header.task_id is required")

    if action == "run-task":
        await _run_task(message, sender, session, runtime, tenant, task_id)
    elif action == "continue-task":
        await _continue_task(message, session, task_id)
    elif action == "finish-task":
        ok = await _finish_task(message, session, task_id)
        runtime.metrics.request("ws", "ok" if ok else "error")
    else:
        raise invalid_parameter(
            "unsupported action; expected run-task, continue-task or finish-task"
        )


async def _run_task(
    message: dict[str, Any],
    sender: WebSocketSender,
    session: WSSession,
    runtime: Any,
    tenant: Tenant,
    task_id: str,
) -> None:
    if not task_id:
        raise invalid_parameter("header.task_id is required for run-task")
    header = message.get("header") or {}
    # "duplex" is the v2 streaming-input flow. The legacy v1 SDK sends
    # "out" for one-shot text (dashscope.audio.tts.SpeechSynthesizer.call):
    # it puts the text in run-task and never sends continue-task or
    # finish-task, so the task is driven to completion below.
    streaming = header.get("streaming")
    if streaming not in (None, "duplex", "out"):
        raise invalid_parameter("only streaming=duplex or out is supported")

    payload = message.get("payload") or {}
    if not isinstance(payload, dict):
        raise invalid_parameter("payload must be a JSON object")
    for key, expected in (
        ("task_group", "audio"),
        ("task", "tts"),
        ("function", "SpeechSynthesizer"),
    ):
        value = payload.get(key)
        if value is not None and value != expected:
            raise invalid_parameter(f"unsupported {key}: {value!r}")

    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise invalid_parameter("payload.model is required")

    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise invalid_parameter("payload.parameters must be a JSON object")
    try:
        options = SpeechInput.model_validate(parameters)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()))
        detail = first.get("msg", "invalid parameters")
        raise invalid_parameter(
            f"invalid parameters.{location}: {detail}" if location else detail
        ) from exc

    request = runtime.build_request(
        model,
        options,
        uuid.uuid4().hex,
        streaming=True,
        allow_empty_text=True,
        api="ws",
    )
    await session.start(task_id, request, tenant)
    runtime.metrics.request("ws", "started")

    if streaming == "out":
        input_ = payload.get("input") or {}
        text = input_.get("text") if isinstance(input_, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise invalid_parameter(
                "payload.input.text is required when streaming=out"
            )
        await session.continue_task(task_id, text)
        # The legacy v1 SDK cannot parse a task-finished payload, so one-shot
        # "out" tasks finish without usage. Duplex tasks keep it.
        completed = await session.finish_task(task_id, include_usage=False)
        runtime.metrics.request("ws", "ok" if completed else "error")


async def _continue_task(
    message: dict[str, Any],
    session: WSSession,
    task_id: str,
) -> None:
    payload = message.get("payload") or {}
    input_ = payload.get("input") if isinstance(payload, dict) else None
    text = input_.get("text") if isinstance(input_, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise invalid_parameter("payload.input.text must be a non-empty string")
    await session.continue_task(task_id, text)


async def _finish_task(
    message: dict[str, Any],
    session: WSSession,
    task_id: str,
) -> bool:
    payload = message.get("payload") or {}
    input_ = payload.get("input") if isinstance(payload, dict) else None
    directive = input_.get("directive") if isinstance(input_, dict) else None
    if directive not in (None, "cancel"):
        raise invalid_parameter(f"unsupported directive: {directive!r}")
    return await session.finish_task(task_id, cancel=directive == "cancel")


def _task_id_of(raw: str) -> str:
    try:
        message = json.loads(raw)
        header = message.get("header") or {}
        task_id = str(header.get("task_id") or "").strip()
        return task_id or _INTERNAL_TASK_ID
    except Exception:  # noqa: BLE001 - best-effort error correlation
        return _INTERNAL_TASK_ID


async def _report_error(
    sender: WebSocketSender,
    session: WSSession,
    task_id: str,
    error: APIError,
) -> None:
    error = normalize_error(error)
    active = session.task
    try:
        if active is not None and active.task_id == task_id:
            # WSSession.fail records the error metric exactly once.
            await session.fail(task_id, error)
        else:
            session.metrics.error(error.code, api="ws")
            await sender.json(bailian.ws_task_failed(task_id, error))
        session.metrics.request("ws", "error")
    except Exception:  # noqa: BLE001 - the peer may already be gone
        logger.debug("could not send task-failed for %s", task_id)
