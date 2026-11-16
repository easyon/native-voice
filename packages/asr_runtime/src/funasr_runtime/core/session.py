from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from cosyvoice_common.errors import APIError, invalid_parameter, timeout_error

from ..core.audio import PCM_BYTES_PER_MS, TextFilter, WavStreamReader
from ..core.lifecycle import Lifecycle
from ..core.limits import TenantLimiter, TenantSlot
from ..core.tenant import Tenant
from ..core.vad import SpeechEnd, SpeechStart, VadStream
from ..engines.base import ASRStream
from ..metrics import Metrics
from ..protocol import bailian
from ..protocol.bailian import RunTask
from ..runtime import Runtime
from ..schemas import StreamParams
from ..settings import ModelConfig

logger = logging.getLogger("funasr_runtime.session")


class Sender(Protocol):
    async def json(self, payload: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class AsrTask:
    task_id: str
    model_key: str
    model: ModelConfig
    params: StreamParams
    text_filter: TextFilter
    tenant: Tenant
    slot: TenantSlot
    vad: VadStream
    heartbeat: bool
    wav_reader: WavStreamReader | None = None

    stream: ASRStream | None = None
    segment_active: bool = False
    segment_begin_ms: int = 0
    segment_had_partial: bool = False
    segment_started_at: float = 0.0
    segment_index: int = 0
    last_partial_text: str = ""
    next_sentence_id: int = 1
    received_bytes: int = 0
    pcm_bytes: int = 0
    pre_roll: bytearray = field(default_factory=bytearray)
    last_emit_at: float = field(default_factory=time.monotonic)
    started_at: float = field(default_factory=time.monotonic)
    heartbeat_task: asyncio.Task[None] | None = None
    finished: bool = False
    released: bool = False


class AsrWSSession:
    """One WebSocket connection; at most one active recognition task."""

    def __init__(
        self,
        *,
        runtime: Runtime,
        sender: Sender,
        tenant: Tenant,
        metrics: Metrics,
        lifecycle: Lifecycle,
        limiter: TenantLimiter,
    ) -> None:
        self.runtime = runtime
        self.sender = sender
        self.tenant = tenant
        self.metrics = metrics
        self.lifecycle = lifecycle
        self.limiter = limiter
        self.task: AsrTask | None = None

    async def run_task(self, message: RunTask) -> None:
        if self.task is not None:
            raise invalid_parameter(
                "a task is already active on this connection; finish it first"
            )
        model_key, model = self.runtime.resolve_model(message.model)
        params = self.runtime.build_params(
            model_key,
            model,
            message.parameters,
            message.context,
        )
        text_filter = TextFilter.parse(message.parameters.special_word_filter)

        await self.lifecycle.enter()
        slot: TenantSlot | None = None
        try:
            slot = await self.limiter.acquire(self.tenant)
            vad = self.runtime.engine.create_vad_stream(model, params)
        except Exception:
            if slot is not None:
                await slot.release()
            await self.lifecycle.leave()
            raise

        reader = (
            WavStreamReader(params.sample_rate)
            if message.parameters.format == "wav"
            else None
        )
        task = AsrTask(
            task_id=message.task_id,
            model_key=model_key,
            model=model,
            params=params,
            text_filter=text_filter,
            tenant=self.tenant,
            slot=slot,
            vad=vad,
            heartbeat=message.parameters.heartbeat,
            wav_reader=reader,
        )
        self.task = task
        self.metrics.request("ws", "started")
        self.metrics.stream_started()
        try:
            await self.sender.json(bailian.ws_task_started(task.task_id))
        except Exception:
            await self._release(task)
            self.task = None
            raise
        if task.heartbeat:
            task.heartbeat_task = asyncio.create_task(self._heartbeat_loop(task))

    async def feed_audio(self, data: bytes) -> None:
        task = self.task
        if task is None:
            raise invalid_parameter("binary audio received before run-task")
        if task.finished:
            raise invalid_parameter("task is already finished")

        elapsed = time.monotonic() - task.started_at
        if elapsed > self.runtime.config.limits.max_task_duration_s:
            raise timeout_error("task exceeded max_task_duration_s")

        task.received_bytes += len(data)
        if task.received_bytes > self.runtime.config.limits.max_audio_bytes_per_task:
            raise APIError(
                "AudioTooLong",
                "per-task audio byte limit exceeded",
                413,
            )

        pipeline = data
        if task.wav_reader is not None:
            pipeline = task.wav_reader.feed(data)
        if not pipeline:
            return
        if len(pipeline) % 2:
            raise invalid_parameter(
                "PCM frames must contain complete 16-bit samples"
            )
        chunk_start_ms = task.pcm_bytes // PCM_BYTES_PER_MS
        task.pcm_bytes += len(pipeline)
        self.runtime.limiter.consume_audio(
            task.tenant, len(pipeline) / (PCM_BYTES_PER_MS * 1000)
        )
        await self._process(task, pipeline, chunk_start_ms)

    async def finish_task(self, task_id: str) -> None:
        task = self.task
        if task is None:
            raise invalid_parameter("finish-task received without an active task")
        if task.task_id != task_id:
            raise invalid_parameter(
                "finish-task task_id does not match the active task"
            )
        if task.finished:
            return
        task.finished = True
        await self._cancel_heartbeat(task)

        if task.wav_reader is not None:
            task.wav_reader.finish()
        timeout = max(0.1, float(self.runtime.config.limits.finish_timeout_s))
        try:
            events = await asyncio.wait_for(task.vad.finish(), timeout=timeout)
            await asyncio.wait_for(
                self._handle_events(
                    task,
                    b"",
                    events,
                    task.pcm_bytes // PCM_BYTES_PER_MS,
                    enforce_limit=False,
                ),
                timeout=timeout,
            )
            if task.segment_active:
                await asyncio.wait_for(
                    self._finish_segment(
                        task, task.pcm_bytes // PCM_BYTES_PER_MS
                    ),
                    timeout=timeout,
                )
        except asyncio.TimeoutError as exc:
            raise timeout_error("finish-task timed out") from exc

        await self.sender.json(bailian.ws_task_finished(task.task_id))
        self.metrics.request("ws", "ok")
        self.metrics.audio_seconds(task.pcm_bytes / (PCM_BYTES_PER_MS * 1000))
        await self._release(task)
        self.task = None

    async def fail(self, task_id: str, error: APIError) -> None:
        task = self.task
        if task is not None and task.task_id == task_id:
            await self._cancel_heartbeat(task)
            await self._release(task)
            self.task = None
        self.metrics.error(error.code, api="ws")
        self.metrics.request("ws", "error")
        await self.sender.json(bailian.ws_task_failed(task_id, error))

    async def close(self) -> None:
        task = self.task
        self.task = None
        if task is None:
            return
        await self._cancel_heartbeat(task)
        await self._release(task)

    def task_expired(self) -> bool:
        task = self.task
        if task is None:
            return False
        return (
            time.monotonic() - task.started_at
            > self.runtime.config.limits.max_task_duration_s
        )

    async def _process(
        self, task: AsrTask, pcm: bytes, chunk_start_ms: int
    ) -> None:
        if not task.segment_active:
            self._append_pre_roll(task, pcm)
        events = await task.vad.feed(pcm)
        await self._handle_events(task, pcm, events, chunk_start_ms)

    async def _handle_events(
        self,
        task: AsrTask,
        pcm: bytes,
        events: list[Any],
        chunk_start_ms: int = 0,
        *,
        enforce_limit: bool = True,
    ) -> None:
        fed = False
        if task.segment_active and pcm:
            await self._feed_stream(task, pcm)
            fed = True

        for event in events:
            if isinstance(event, SpeechStart):
                if task.segment_active:
                    continue
                self._start_segment(task, event.time_ms)
                if not fed:
                    if task.pre_roll:
                        await self._feed_stream(task, bytes(task.pre_roll))
                    else:
                        tail = pcm[
                            max(
                                0,
                                (task.segment_begin_ms - chunk_start_ms)
                                * PCM_BYTES_PER_MS,
                            ) :
                        ]
                        if tail:
                            await self._feed_stream(task, tail)
                    fed = True
                elif pcm:
                    # The chunk was already fed to the previous segment, but
                    # it may also contain the start of this one after a
                    # same-chunk VAD boundary.
                    tail = pcm[
                        max(
                            0,
                            (task.segment_begin_ms - chunk_start_ms)
                            * PCM_BYTES_PER_MS,
                        ) :
                    ]
                    if tail:
                        await self._feed_stream(task, tail)
                task.pre_roll.clear()
            elif isinstance(event, SpeechEnd):
                if not task.segment_active:
                    continue
                if not fed and pcm:
                    await self._feed_stream(task, pcm)
                    fed = True
                await self._finish_segment(task, int(event.time_ms))

        if task.segment_active and not fed and pcm:
            await self._feed_stream(task, pcm)
        if enforce_limit:
            await self._enforce_segment_limit(task)

    def _start_segment(self, task: AsrTask, begin_ms: int) -> None:
        task.segment_active = True
        task.segment_begin_ms = max(0, int(begin_ms))
        task.segment_had_partial = False
        task.last_partial_text = ""
        task.segment_started_at = time.monotonic()
        task.stream = self.runtime.engine.create_stream(
            task.model,
            replace(
                task.params,
                extra={
                    **task.params.extra,
                    "begin_time_ms": task.segment_begin_ms,
                    "segment_index": task.segment_index,
                },
            ),
        )

    async def _enforce_segment_limit(self, task: AsrTask) -> None:
        limit_ms = int(task.params.max_segment_ms)
        if limit_ms <= 0:
            return
        while task.segment_active:
            boundary = task.pcm_bytes // PCM_BYTES_PER_MS
            if boundary - task.segment_begin_ms < limit_ms:
                return
            await self._finish_segment(task, boundary)
            # Speech may continue past the cap. Start a fresh segment at the
            # same boundary so the next chunk is decoded instead of growing an
            # unbounded buffer (and an ever slower final decode).
            self._start_segment(task, boundary)

    async def _feed_stream(self, task: AsrTask, pcm: bytes) -> None:
        stream = task.stream
        if stream is None:
            return
        partials = await stream.feed(pcm)
        for partial in partials:
            await self._emit_partial(task, partial)

    async def _finish_segment(self, task: AsrTask, end_ms: int) -> None:
        stream = task.stream
        task.stream = None
        final = None
        if stream is not None:
            final = await stream.finish()
            await stream.close()
        await self._emit_final(task, final, end_ms)
        task.segment_active = False
        task.segment_had_partial = False
        task.last_partial_text = ""
        task.pre_roll.clear()
        task.segment_index += 1

    async def _emit_partial(self, task: AsrTask, partial: Any) -> None:
        text = task.text_filter.apply(partial.text).strip()
        if not text or text == task.last_partial_text:
            return
        first = not task.segment_had_partial
        task.segment_had_partial = True
        task.last_partial_text = text
        task.last_emit_at = time.monotonic()
        await self.sender.json(
            bailian.ws_sentence(
                task.task_id,
                sentence_id=task.next_sentence_id,
                begin_time=task.segment_begin_ms,
                end_time=None,
                text=text,
                sentence_begin=first,
                sentence_end=False,
                usage=None,
            )
        )
        self.metrics.inc("funasr_partials_total")
        if first:
            self.metrics.first_partial(
                time.monotonic() - task.segment_started_at,
                model=task.model_key,
            )

    async def _emit_final(
        self, task: AsrTask, final: Any, end_ms: int
    ) -> None:
        if final is None:
            return
        text = task.text_filter.apply(final.text).strip()
        if not text:
            return
        begin = task.segment_begin_ms
        end = max(
            begin,
            int(end_ms),
            int(getattr(final, "end_time", begin)),
        )
        usage = {"duration": max(1, round(task.pcm_bytes / (PCM_BYTES_PER_MS * 1000)))}
        await self.sender.json(
            bailian.ws_sentence(
                task.task_id,
                sentence_id=task.next_sentence_id,
                begin_time=begin,
                end_time=end,
                text=text,
                sentence_begin=not task.segment_had_partial,
                sentence_end=True,
                words=bailian.words_payload(final.words),
                usage=usage,
            )
        )
        task.next_sentence_id += 1
        task.last_emit_at = time.monotonic()
        self.metrics.sentence()
        duration_s = max(0.001, (end - begin) / 1000.0)
        self.metrics.final_latency(
            time.monotonic() - task.segment_started_at,
            model=task.model_key,
        )
        self.metrics.rtf(
            (time.monotonic() - task.segment_started_at) / duration_s,
            model=task.model_key,
        )

    def _append_pre_roll(self, task: AsrTask, pcm: bytes) -> None:
        maximum = max(
            0,
            self.runtime.config.engine.vad.pre_roll_ms * PCM_BYTES_PER_MS,
        )
        if maximum <= 0:
            return
        task.pre_roll.extend(pcm)
        if len(task.pre_roll) > maximum:
            del task.pre_roll[: len(task.pre_roll) - maximum]

    async def _heartbeat_loop(self, task: AsrTask) -> None:
        interval = max(
            0.1,
            self.runtime.config.limits.heartbeat_interval_ms / 1000.0,
        )
        try:
            while True:
                await asyncio.sleep(interval)
                if task.finished or self.task is not task:
                    return
                now = time.monotonic()
                if now - task.last_emit_at < interval:
                    continue
                await self.sender.json(
                    bailian.ws_sentence(
                        task.task_id,
                        sentence_id=0,
                        begin_time=task.pcm_bytes // PCM_BYTES_PER_MS,
                        end_time=None,
                        text="",
                        sentence_begin=False,
                        sentence_end=False,
                        heartbeat=True,
                        usage=None,
                    )
                )
                task.last_emit_at = now
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the peer may already be gone
            logger.debug("heartbeat stopped for task %s", task.task_id)

    async def _cancel_heartbeat(self, task: AsrTask) -> None:
        heartbeat = task.heartbeat_task
        task.heartbeat_task = None
        if heartbeat is None:
            return
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat

    async def _release(self, task: AsrTask) -> None:
        if task.released:
            return
        task.released = True
        if task.stream is not None:
            await task.stream.close()
            task.stream = None
        await task.vad.close()
        await task.slot.release()
        await self.lifecycle.leave()
        self.metrics.stream_finished()
