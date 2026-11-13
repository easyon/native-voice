from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from cosyvoice_common.errors import (
    APIError,
    internal_error,
    service_unavailable,
    throttled,
    timeout_error,
)
from cosyvoice_common.text import count_characters, split_sentences

from ..engines.base import AudioChunk, TTSEngine
from ..metrics import Metrics
from ..schemas import SynthesisRequest
from ..settings import LimitsConfig
from .timeout import deadline

logger = logging.getLogger("cosyvoice_runtime.orchestrator")


def _map_engine_error(exc: Exception) -> APIError:
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    if "outofmemory" in name or "out of memory" in text or "cuda" in name:
        return service_unavailable("engine resource exhausted")
    return internal_error("synthesis failed")


class SynthesisOrchestrator:
    """Transport-agnostic synthesis flow.

    HTTP, SSE and WebSocket adapters all use this class; they only differ in how
    they frame the resulting chunks. This keeps model/engine concerns out of the
    protocol layer and makes it straightforward to swap the engine.
    """

    def __init__(
        self,
        engine: TTSEngine,
        limits: LimitsConfig,
        metrics: Metrics,
    ) -> None:
        self.engine = engine
        self.limits = limits
        self.metrics = metrics
        self._semaphore = asyncio.Semaphore(max(1, int(limits.max_concurrent_streams)))
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def active_streams(self) -> int:
        return self._active

    @asynccontextmanager
    async def slot(self, api: str = "http") -> AsyncIterator[None]:
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=max(0.1, float(self.limits.acquire_timeout_s)),
            )
        except asyncio.TimeoutError as exc:
            raise throttled() from exc

        self.metrics.queue_wait(time.monotonic() - started, api=api)
        self._active += 1
        self._idle.clear()
        self.metrics.stream_started()
        try:
            yield
        finally:
            self.metrics.stream_finished()
            self._active = max(0, self._active - 1)
            if self._active == 0:
                self._idle.set()
            self._semaphore.release()

    async def wait_idle(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=max(0.1, timeout))
            return True
        except asyncio.TimeoutError:
            return False

    def sentences(self, request: SynthesisRequest) -> list[str]:
        return split_sentences(request.text)

    async def synthesize_full(
        self, request: SynthesisRequest
    ) -> tuple[bytes, int]:
        started = time.monotonic()
        pcm = bytearray()
        source_rate = request.sample_rate
        stream = self.engine.synthesize(request)
        try:
            async with self.slot(request.api):
                async with deadline(self.limits.sentence_timeout_s):
                    async for chunk in stream:
                        source_rate = chunk.sample_rate
                        pcm.extend(chunk.pcm)
        except APIError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - map engine failures
            logger.exception(
                "engine synthesis failed: request_id=%s model=%s voice=%s",
                request.request_id,
                request.model,
                request.voice,
            )
            raise _map_engine_error(exc) from exc
        finally:
            await stream.aclose()

        elapsed = time.monotonic() - started
        audio_seconds = len(pcm) / max(1, source_rate * 2)
        self.metrics.sentence()
        self.metrics.audio(len(pcm))
        self.metrics.request_duration(elapsed, api=request.api, model=request.model)
        if audio_seconds > 0:
            self.metrics.rtf(elapsed / audio_seconds, model=request.model)
        return bytes(pcm), source_rate

    async def stream_sentence(
        self,
        request: SynthesisRequest,
        sentence: str,
    ) -> AsyncIterator[AudioChunk]:
        from dataclasses import replace

        sentence_request = replace(request, text=sentence, streaming=True)
        started = time.monotonic()
        first_chunk_at: float | None = None
        total_bytes = 0
        source_rate = request.sample_rate
        stream = self.engine.synthesize(sentence_request)
        iterator = stream.__aiter__()
        try:
            async with self.slot(request.api):
                async with deadline(self.limits.sentence_timeout_s):
                    while True:
                        timeout = (
                            self.limits.first_audio_timeout_s
                            if first_chunk_at is None
                            else None
                        )
                        try:
                            if timeout is None:
                                chunk = await iterator.__anext__()
                            else:
                                chunk = await asyncio.wait_for(
                                    iterator.__anext__(),
                                    timeout=max(0.1, float(timeout)),
                                )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as exc:
                            raise timeout_error("first audio timed out") from exc

                        source_rate = chunk.sample_rate
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                            self.metrics.first_audio(
                                first_chunk_at - started,
                                api=request.api,
                                model=request.model,
                            )
                        total_bytes += len(chunk.pcm)

                        # CosyVoice yields one whole sentence at a time. On a
                        # slow proxy/tunnel the client otherwise waits for the
                        # entire sentence payload before it can start playback.
                        # PCM/WAV concatenate safely; MP3 is left intact because
                        # independently encoded chunks add encoder padding.
                        limit = int(self.limits.stream_chunk_bytes)
                        if (
                            limit > 0
                            and request.format in {"pcm", "wav"}
                            and len(chunk.pcm) > limit
                        ):
                            for offset in range(0, len(chunk.pcm), limit):
                                yield AudioChunk(
                                    pcm=chunk.pcm[offset : offset + limit],
                                    sample_rate=chunk.sample_rate,
                                )
                        else:
                            yield chunk
        except APIError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - map engine failures
            logger.exception(
                "engine stream failed: request_id=%s model=%s voice=%s",
                request.request_id,
                request.model,
                request.voice,
            )
            raise _map_engine_error(exc) from exc
        finally:
            await stream.aclose()

        elapsed = time.monotonic() - started
        audio_seconds = total_bytes / max(1, source_rate * 2)
        self.metrics.sentence()
        self.metrics.sentence_duration(elapsed, api=request.api)
        if audio_seconds > 0:
            self.metrics.rtf(elapsed / audio_seconds, api=request.api, model=request.model)

    def characters(self, request: SynthesisRequest) -> int:
        return count_characters(request.text)
