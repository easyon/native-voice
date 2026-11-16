from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from cosyvoice_common.errors import (
    APIError,
    invalid_parameter,
    normalize_error,
    timeout_error,
)
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..context import bind_context
from ..core.session import AsrWSSession
from ..protocol import bailian
from ..protocol.bailian import FinishTask, RunTask, parse_client_message
from .deps import authenticate, runtime_from_websocket

logger = logging.getLogger("funasr_runtime.ws")
router = APIRouter()

_INTERNAL_TASK_ID = "00000000-0000-0000-0000-000000000000"


class WebSocketSender:
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
    session = AsrWSSession(
        runtime=runtime,
        sender=sender,
        tenant=tenant,
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
                if session.task is not None:
                    if session.task_expired():
                        await _report_and_close(
                            websocket,
                            sender,
                            session,
                            _active_task_id(session),
                            timeout_error("task exceeded max_task_duration_s"),
                        )
                        break
                    continue
                logger.info("websocket idle timeout: tenant=%s", tenant.id)
                break

            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                if len(data) > max_message:
                    await _report_and_close(
                        websocket,
                        sender,
                        session,
                        _active_task_id(session),
                        invalid_parameter("websocket binary frame is too large"),
                    )
                    break
                task_id = _active_task_id(session)
                try:
                    with bind_context(
                        request_id=task_id,
                        tenant=tenant.id,
                        task_id=task_id,
                        api="ws",
                    ):
                        await session.feed_audio(data)
                except APIError as exc:
                    await _report_and_close(
                        websocket, sender, session, task_id, exc
                    )
                    break
                continue

            raw = message.get("text")
            if raw is None:
                continue
            if len(raw.encode("utf-8")) > max_message:
                await _report_and_close(
                    websocket,
                    sender,
                    session,
                    _task_id_of(raw),
                    invalid_parameter("websocket message is too large"),
                )
                break

            task_id = _task_id_of(raw)
            try:
                with bind_context(
                    request_id=task_id,
                    tenant=tenant.id,
                    task_id=task_id,
                    api="ws",
                ):
                    client_message = parse_client_message(raw)
                    if isinstance(client_message, RunTask):
                        await session.run_task(client_message)
                    elif isinstance(client_message, FinishTask):
                        await session.finish_task(client_message.task_id)
            except APIError as exc:
                await _report_and_close(
                    websocket, sender, session, task_id, exc
                )
                break
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - keep a bad connection from killing the worker
        logger.exception("websocket connection failed")
    finally:
        await session.close()
        await runtime.websocket_disconnected()
        logger.info("websocket disconnected: tenant=%s", tenant.id)


async def _report_and_close(
    websocket: WebSocket,
    sender: WebSocketSender,
    session: AsrWSSession,
    task_id: str,
    error: APIError,
) -> None:
    error = normalize_error(error)
    active = session.task
    try:
        if active is not None and active.task_id == task_id:
            await session.fail(task_id, error)
        else:
            session.metrics.error(error.code, api="ws")
            session.metrics.request("ws", "error")
            await sender.json(bailian.ws_task_failed(task_id, error))
    except Exception:  # noqa: BLE001 - the peer may already be gone
        logger.debug("could not send task-failed for %s", task_id)

    close_code = 1008 if error.status_code < 500 else 1011
    with contextlib.suppress(Exception):
        await websocket.close(code=close_code)


def _active_task_id(session: AsrWSSession) -> str:
    task = session.task
    return task.task_id if task is not None else _INTERNAL_TASK_ID


def _task_id_of(raw: str) -> str:
    try:
        message = json.loads(raw)
        header = message.get("header") or {}
        task_id = str(header.get("task_id") or "").strip()
        return task_id or _INTERNAL_TASK_ID
    except Exception:  # noqa: BLE001 - best-effort error correlation
        return _INTERNAL_TASK_ID
