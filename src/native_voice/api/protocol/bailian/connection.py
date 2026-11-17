"""Per-connection task state machine for the Bailian-compatible endpoint."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ....backends.base import Backends
from ....config.schema import ServiceConfig
from ....observability.metrics import METRICS
from ....security.auth import Authenticator, AuthError, Principal
from ....security.quotas import QuotaExceeded, QuotaManager
from ..asr.handler import ASRTaskHandler
from ..tts.handler import TTSTaskHandler
from .errors import ProtocolError, TaskError
from .events import ClientEvent, parse_client_event, task_failed, task_finished, task_started
from .writer import ConnectionWriter

log = logging.getLogger(__name__)


class BailianConnection:
    def __init__(
        self,
        websocket,
        principal: Principal,
        cfg: ServiceConfig,
        backends: Backends,
        quotas: QuotaManager,
        authenticator: Authenticator,
        forced_kind: str | None = None,
    ):
        self.websocket = websocket
        self.principal = principal
        self.cfg = cfg
        self.backends = backends
        self.quotas = quotas
        self.authenticator = authenticator
        self.forced_kind = forced_kind
        self.writer: ConnectionWriter | None = None
        self.handler = None
        self.task_id: str | None = None
        self.kind: str | None = None
        self.lease = None
        self.active = False

    async def run(self) -> None:
        await self.websocket.accept()
        self.writer = ConnectionWriter(self.websocket)
        METRICS.ws_connections.labels("inference").inc()
        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.receive(), timeout=self.cfg.server.ws_idle_timeout_s
                    )
                except asyncio.TimeoutError:
                    await self._close(1001, "idle timeout")
                    return
                if message.get("type") == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    await self._on_bytes(message["bytes"])
                    continue
                text = message.get("text")
                if text is not None:
                    await self._on_text(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - connection-level
            log.debug("connection loop ended: %r", exc)
        finally:
            await self._cleanup()

    # --------------------------------------------------------------- messages
    async def _on_text(self, text: str) -> None:
        try:
            ev = parse_client_event(text)
        except ProtocolError as exc:
            if self.active:
                await self._fail("InvalidParameter", str(exc))
            else:
                await self._close(1008, str(exc))
            return

        action = ev.header.action
        if action == "run-task":
            await self._start(ev)
        elif action == "continue-task":
            await self._continue(ev)
        elif action == "finish-task":
            await self._finish(ev)
        else:
            await self._close(1008, f"unknown action: {action}")

    async def _on_bytes(self, data: bytes) -> None:
        if not self.active or self.kind != "asr":
            return
        if len(data) > self.cfg.server.max_message_bytes:
            await self._fail("InvalidParameter", "audio frame too large")
            return
        try:
            await self.handler.on_audio(data)
        except TaskError as exc:
            await self._fail(exc.code, exc.message)
        except Exception as exc:  # pragma: no cover
            await self._fail("InternalError", repr(exc))

    # ------------------------------------------------------------------ tasks
    async def _start(self, ev: ClientEvent) -> None:
        payload: dict[str, Any] = ev.payload or {}
        task = str(payload.get("task", ""))
        task_id = ev.header.task_id or "-"
        if self.forced_kind and task and task != self.forced_kind:
            await self.writer.send_json(
                task_failed(task_id, "InvalidParameter", f"task must be {self.forced_kind}")
            )
            return
        if self.active:
            await self.writer.send_json(
                task_failed(task_id, "TaskAlreadyRunning", "a task is already running")
            )
            return
        if task not in ("asr", "tts"):
            await self.writer.send_json(task_failed(task_id, "InvalidParameter", f"unsupported task: {task}"))
            return
        try:
            self.authenticator.require(self.principal, task)
            self.quotas.check_daily(self.principal, task)
        except AuthError as exc:
            await self.writer.send_json(task_failed(task_id, "Forbidden", str(exc)))
            return
        except QuotaExceeded as exc:
            await self.writer.send_json(task_failed(task_id, "Throttling", str(exc)))
            return
        try:
            lease = self.quotas.acquire(self.principal)
        except QuotaExceeded as exc:
            await self.writer.send_json(task_failed(task_id, "Throttling", str(exc)))
            return

        params = payload.get("parameters") or {}
        try:
            if task == "asr":
                handler = ASRTaskHandler(task_id, params, self.backends.asr, self.writer, self.cfg)
            else:
                handler = TTSTaskHandler(task_id, params, self.backends.tts, self.writer, self.cfg)
        except TaskError as exc:
            self.quotas.release(lease)
            await self.writer.send_json(task_failed(task_id, exc.code, exc.message))
            return
        self.handler, self.task_id, self.kind, self.lease = handler, task_id, task, lease
        self.active = True
        await self.writer.send_json(task_started(task_id))
        if task == "tts":
            await handler.start()
        log.info("task started task_id=%s kind=%s", task_id, task)

    async def _continue(self, ev: ClientEvent) -> None:
        if not self.active or self.kind != "tts":
            await self._close(1008, "continue-task without a TTS task")
            return
        try:
            await self.handler.on_continue(ev.payload or {})
        except TaskError as exc:
            await self._fail(exc.code, exc.message)
        except Exception as exc:
            await self._fail("InternalError", repr(exc))

    async def _finish(self, ev: ClientEvent) -> None:
        if not self.active:
            await self._close(1008, "finish-task without a running task")
            return
        try:
            usage = await self.handler.on_finish(ev.payload or {})
        except TaskError as exc:
            await self._fail(exc.code, exc.message)
            return
        except Exception as exc:
            await self._fail("InternalError", repr(exc))
            return
        await self.writer.send_json(task_finished(self.task_id or "-", usage))
        METRICS.tasks_total.labels(self.kind or "unknown", "success").inc()
        self._release()

    async def _fail(self, code: str, message: str) -> None:
        METRICS.errors_total.labels(code).inc()
        if self.active and self.writer is not None:
            await self.writer.send_json(task_failed(self.task_id or "-", code, message))
            await self.handler.cancel()
            METRICS.tasks_total.labels(self.kind or "unknown", "failed").inc()
        self._release()

    def _release(self) -> None:
        if self.lease is not None:
            self.quotas.release(self.lease)
        self.active = False
        self.handler = None
        self.lease = None
        self.task_id = None
        self.kind = None

    async def _cleanup(self) -> None:
        try:
            if self.active and self.handler is not None:
                try:
                    await self.handler.cancel()
                except Exception:
                    pass
            self._release()
            if self.writer is not None:
                await self.writer.close()
        finally:
            METRICS.ws_connections.labels("inference").dec()

    async def _close(self, code: int, reason: str) -> None:
        try:
            await self.websocket.close(code=code, reason=reason)
        except Exception:
            pass
