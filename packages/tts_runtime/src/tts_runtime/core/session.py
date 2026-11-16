from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace
from typing import Any, Protocol

from cosyvoice_common.audio import encode_stream_chunk_async
from cosyvoice_common.errors import (
    APIError,
    internal_error,
    invalid_parameter,
    text_too_long,
    throttled,
    timeout_error,
    unsupported_parameter,
)
from cosyvoice_common.text import SentenceBuffer, count_characters

from ..context import bind_context
from ..metrics import Metrics
from ..protocol import bailian
from ..schemas import SynthesisRequest
from ..settings import LimitsConfig
from .lifecycle import Lifecycle
from .limits import TenantLimiter, TenantSlot
from .orchestrator import SynthesisOrchestrator
from .sentence_pipeline import (
    SentenceAudio,
    SentenceDone,
    SentenceStart,
    stream_sentences,
)
from .tenant import Tenant

logger = logging.getLogger("tts_runtime.ws")


class Sender(Protocol):
    async def json(self, payload: dict[str, Any]) -> None: ...

    async def bytes(self, data: bytes) -> None: ...

    async def audio(self, payload: dict[str, Any], data: bytes) -> None:
        """Send the JSON metadata and its binary frame atomically."""
        ...


@dataclass(slots=True)
class WSTask:
    task_id: str
    request: SynthesisRequest
    queue: asyncio.Queue[tuple[str, str | None]]
    tenant: Tenant
    slot: TenantSlot
    received_chars: int = 0
    synthesized_chars: int = 0
    synthesized_ms: int = 0
    sentence_index: int = 0
    ssml_seen: bool = False
    audio_started: bool = False
    error: APIError | None = None
    failed_sent: bool = False
    finished: bool = False
    cancelled: bool = False
    released: bool = False
    started_at: float = 0.0
    deadline_task: asyncio.Task[None] | None = None


class WSSession:
    """WebSocket task lifecycle, independent of the Starlette transport.

    Reliability properties:
    - one active task per connection;
    - bounded text queue (backpressure -> Throttling);
    - cancel closes the engine generator and releases the GPU slot;
    - errors are sent once as task-failed;
    - max task duration and graceful drain are enforced here;
    - the connection can be reused after finish/cancel.
    """

    def __init__(
        self,
        orchestrator: SynthesisOrchestrator,
        sender: Sender,
        limits: LimitsConfig,
        metrics: Metrics,
        lifecycle: Lifecycle,
        limiter: TenantLimiter,
    ) -> None:
        self.orchestrator = orchestrator
        self.sender = sender
        self.limits = limits
        self.metrics = metrics
        self.lifecycle = lifecycle
        self.limiter = limiter
        self._task: WSTask | None = None
        self._runner: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def task(self) -> WSTask | None:
        return self._task

    async def start(
        self,
        task_id: str,
        request: SynthesisRequest,
        tenant: Tenant,
    ) -> None:
        if self._closed:
            raise invalid_parameter("websocket session is closed")
        if self._runner is not None and not self._runner.done():
            raise invalid_parameter("a task is already running on this connection")

        await self.lifecycle.enter()
        try:
            slot = await self.limiter.acquire(tenant)
        except Exception:
            await self.lifecycle.leave()
            raise

        task = WSTask(
            task_id=task_id,
            request=request,
            queue=asyncio.Queue(maxsize=max(1, int(self.limits.ws_queue_size))),
            tenant=tenant,
            slot=slot,
            started_at=time.monotonic(),
        )
        self._task = task
        try:
            await self.sender.json(bailian.ws_task_started(task_id))
        except Exception:
            await self._release(task)
            self._task = None
            raise

        self._runner = asyncio.create_task(
            self._run(task),
            name=f"ws-task-{task_id[:8]}",
        )
        task.deadline_task = asyncio.create_task(
            self._deadline(task),
            name=f"ws-deadline-{task_id[:8]}",
        )

    async def continue_task(self, task_id: str, text: str) -> None:
        task = self._require_task(task_id)
        if task.finished:
            raise invalid_parameter("task is already finished")
        if task.error is not None:
            raise task.error
        if len(text) > self.limits.max_text_chars:
            raise text_too_long(
                f"continue-task text length {len(text)} exceeds "
                f"{self.limits.max_text_chars}"
            )
        if task.request.enable_ssml and task.ssml_seen:
            raise unsupported_parameter(
                "only one continue-task is allowed when enable_ssml=true"
            )
        if task.request.enable_ssml:
            task.ssml_seen = True
        received = count_characters(text)
        if task.received_chars + received > self.limits.max_total_chars:
            raise text_too_long(
                f"total text exceeds {self.limits.max_total_chars} characters"
            )
        self.limiter.consume_characters(task.tenant, received)
        task.received_chars += received
        await self._put(task, ("text", text))

    async def finish_task(
        self,
        task_id: str,
        *,
        cancel: bool = False,
        include_usage: bool = True,
    ) -> bool:
        """Finish a task and return ``True`` when it completed successfully."""
        task = self._require_task(task_id)
        if task.finished:
            raise invalid_parameter("task is already finished")
        runner = self._runner
        characters = task.synthesized_chars if include_usage else None

        if cancel:
            await self._cancel(task)
            task.finished = True
            await self._release(task)
            if task.error is None:
                await self.sender.json(
                    bailian.ws_task_finished(
                        task.task_id,
                        task.request.request_id,
                        characters,
                    )
                )
        else:
            await self._put(task, ("flush", None))
            if runner is not None:
                done, _ = await asyncio.wait(
                    {runner},
                    timeout=max(1.0, float(self.limits.max_task_duration_s)),
                )
                if not done:
                    task.error = timeout_error("task exceeded maximum duration")
                    await self._cancel(task)
            task.finished = True
            await self._release(task)
            if task.error is None:
                await self.sender.json(
                    bailian.ws_task_finished(
                        task.task_id,
                        task.request.request_id,
                        characters,
                    )
                )
            elif not task.failed_sent:
                await self._safe_fail(task, task.error)
        self._task = None
        self._detach_runner()
        return task.error is None

    async def fail(self, task_id: str, error: APIError) -> None:
        """Terminate the active task with a single ``task-failed`` event."""
        task = self._require_task(task_id)
        task.error = error
        await self._cancel(task)
        task.finished = True
        await self._safe_fail(task, error)
        await self._release(task)
        self._task = None
        self._detach_runner()

    async def close(self) -> None:
        self._closed = True
        if self._task is not None:
            await self._cancel(self._task)
            self._task.finished = True
            await self._release(self._task)
        else:
            await self._cancel_runner()
        self._task = None
        self._runner = None

    def _detach_runner(self) -> None:
        """Forget the runner only after it has stopped.

        If cancellation could not stop it within the grace period (a client
        that stopped reading), keeping the handle makes ``start`` reject a new
        task until it exits, so two synthesis loops cannot interleave frames
        on one connection.
        """
        if self._runner is not None and self._runner.done():
            self._runner = None

    async def _run(self, task: WSTask) -> None:
        buffer = SentenceBuffer()
        try:
            with bind_context(
                request_id=task.request.request_id,
                tenant=task.tenant.id,
                task_id=task.task_id,
                model=task.request.model,
                voice=task.request.voice,
                api="ws",
            ):
                while True:
                    kind, value = await task.queue.get()
                    if kind == "cancel" or task.cancelled:
                        return
                    if kind == "text":
                        for sentence in buffer.feed(value or ""):
                            await self._synthesize(task, sentence)
                            if task.cancelled:
                                return
                    elif kind == "flush":
                        for sentence in buffer.flush():
                            await self._synthesize(task, sentence)
                            if task.cancelled:
                                return
                        return
        except asyncio.CancelledError:
            raise
        except APIError as exc:
            task.error = exc
            await self._safe_fail(task, exc)
        except Exception:  # noqa: BLE001 - isolate task failures
            logger.exception("ws task failed: task_id=%s", task.task_id)
            error = internal_error()
            task.error = error
            await self._safe_fail(task, error)

    async def _synthesize(self, task: WSTask, sentence: str) -> None:
        index = task.sentence_index
        task.sentence_index += 1
        start_ms = task.synthesized_ms
        sentence_ms = 0
        words: list[dict[str, Any]] = []
        pipeline = stream_sentences(
            self.orchestrator,
            replace(task.request, text=sentence),
        )
        try:
            async for event in pipeline:
                # Cancellation is cooperative here: cancelling the runner while
                # it is inside asyncio.wait_for can be swallowed by wait_for
                # (the inner operation may complete at the same time), so the
                # flag guarantees no more audio is produced after finish-task
                # directive=cancel.
                if task.cancelled:
                    return
                if isinstance(event, SentenceStart):
                    await self.sender.json(
                        bailian.ws_result(
                            task.task_id,
                            {
                                "type": "sentence-begin",
                                "original_text": event.text,
                                "sentence": {
                                    "index": index,
                                    "words": words,
                                    "begin_time": start_ms,
                                    "end_time": start_ms,
                                },
                            },
                        )
                    )
                elif isinstance(event, SentenceAudio):
                    sentence_ms += int(
                        len(event.chunk.pcm) / (event.chunk.sample_rate * 2) * 1000
                    )
                    encoded = await encode_stream_chunk_async(
                        event.chunk.pcm,
                        event.chunk.sample_rate,
                        task.request.format,
                        sample_rate=task.request.sample_rate,
                        volume=task.request.volume,
                        pitch=task.request.pitch,
                        bit_rate=task.request.bit_rate,
                        first=not task.audio_started,
                    )
                    task.audio_started = True
                    self.metrics.audio(len(encoded))
                    if task.cancelled:
                        return
                    await self.sender.audio(
                        bailian.ws_result(
                            task.task_id,
                            {
                                "type": "sentence-synthesis",
                                "sentence": {
                                    "index": index,
                                    "words": words,
                                    "begin_time": start_ms,
                                    "end_time": start_ms,
                                },
                            },
                        ),
                        encoded,
                    )
                elif isinstance(event, SentenceDone):
                    words = event.words
                    task.synthesized_chars += event.characters
                    task.synthesized_ms = start_ms + sentence_ms
                    await self.sender.json(
                        bailian.ws_result(
                            task.task_id,
                            {
                                "type": "sentence-end",
                                "original_text": event.text,
                                "sentence": {
                                    "index": index,
                                    "words": words,
                                    "begin_time": start_ms,
                                    "end_time": task.synthesized_ms,
                                },
                            },
                            usage={"characters": task.synthesized_chars},
                        )
                    )
        finally:
            await pipeline.aclose()

    async def _deadline(self, task: WSTask) -> None:
        try:
            await asyncio.sleep(max(1.0, float(self.limits.max_task_duration_s)))
        except asyncio.CancelledError:
            return
        if self._task is not task or task.finished:
            return
        task.error = timeout_error("task exceeded maximum duration")
        await self._cancel(task)
        task.finished = True
        await self._safe_fail(task, task.error)
        await self._release(task)
        self._task = None
        self._detach_runner()

    async def _put(self, task: WSTask, item: tuple[str, str | None]) -> None:
        try:
            await asyncio.wait_for(task.queue.put(item), timeout=1.0)
        except asyncio.TimeoutError as exc:
            raise throttled("websocket task queue is full") from exc

    def _require_task(self, task_id: str) -> WSTask:
        if self._task is None:
            raise invalid_parameter("no active task; send run-task first")
        if self._task.task_id != task_id:
            raise invalid_parameter(
                f"task_id mismatch: expected {self._task.task_id}, got {task_id}"
            )
        return self._task

    async def _cancel(self, task: WSTask) -> None:
        """Stop the runner cooperatively, then cancellably.

        ``Task.cancel()`` alone is not sufficient: when the runner is suspended
        inside ``asyncio.wait_for`` (every websocket send), wait_for may return
        the inner result instead of propagating the cancellation if the inner
        operation completes at the same time. The flag and queue sentinel make
        the stop deterministic, so a cancelled task cannot keep producing audio
        and ``finish-task`` cannot hang on a slow client.
        """
        task.cancelled = True
        try:
            task.queue.put_nowait(("cancel", None))
        except asyncio.QueueFull:
            # The runner is busy with the current sentence; the flag stops it
            # before the next one.
            pass
        await self._cancel_runner()

    async def _cancel_runner(self) -> None:
        runner = self._runner
        if runner is None or runner.done():
            return
        runner.cancel()
        grace = max(0.1, float(self.limits.ws_cancel_timeout_s))
        done, _ = await asyncio.wait({runner}, timeout=grace)
        if not done:
            logger.warning(
                "ws task runner did not stop within %.1fs of cancellation",
                grace,
            )
            return
        if not runner.cancelled():
            exc = runner.exception()
            if exc is not None:
                logger.debug(
                    "ws task runner ended with %s while being cancelled",
                    type(exc).__name__,
                )

    async def _release(self, task: WSTask) -> None:
        if task.released:
            return
        task.released = True
        if task.deadline_task is not None:
            if task.deadline_task is not asyncio.current_task():
                task.deadline_task.cancel()
            task.deadline_task = None
        await task.slot.release()
        await self.lifecycle.leave()

    async def _safe_fail(self, task: WSTask, error: APIError) -> None:
        if task.failed_sent:
            return
        task.failed_sent = True
        self.metrics.error(error.code, api="ws")
        try:
            await self.sender.json(bailian.ws_task_failed(task.task_id, error))
        except Exception:  # client may already be gone
            logger.debug("could not send task-failed for %s", task.task_id)
