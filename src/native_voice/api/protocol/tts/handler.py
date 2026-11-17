"""Bailian-compatible TTS task: run-task -> continue-task -> finish-task."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import numpy as np

from ....audio.encode import AudioFormatError, encode_audio, is_streamable
from ....backends.base import TTSBackend
from ....config.schema import ServiceConfig
from ....observability.metrics import METRICS
from ..bailian.errors import TaskError
from ..bailian.events import tts_result
from ..bailian.writer import ConnectionWriter
from .sentences import TTSSentenceBuffer

log = logging.getLogger(__name__)
MAX_SINGLE_CHUNK = 20_000
MAX_TOTAL_CHARS = 200_000
MAX_CONTINUE_GAP_S = 23.0


class TTSTaskHandler:
    def __init__(
        self,
        task_id: str,
        params: dict[str, Any],
        backend: TTSBackend,
        writer: ConnectionWriter,
        cfg: ServiceConfig,
    ):
        self.task_id = task_id
        self.backend = backend
        self.writer = writer
        self.cfg = cfg
        self.text_type = str(params.get("text_type", "PlainText"))
        if self.text_type != "PlainText":
            raise TaskError("UnsupportedFeature", f"text_type '{self.text_type}' is not supported")
        if bool(params.get("enable_ssml", False)):
            raise TaskError("UnsupportedFeature", "SSML is not supported by the local engine")
        if float(params.get("rate", 1.0)) != 1.0:
            raise TaskError("UnsupportedFeature", "rate != 1.0 is not supported by the streaming engine")
        if float(params.get("pitch", 1.0)) != 1.0:
            raise TaskError("UnsupportedFeature", "pitch is not supported by the local engine")

        requested_voice = str(params.get("voice", "default"))
        self.voice = cfg.tts.voice_aliases.get(requested_voice, requested_voice)
        known = {v["name"] for v in backend.voices()}
        if self.voice not in known:
            raise TaskError("InvalidParameter", f"unknown voice: {requested_voice}")

        self.fmt = str(params.get("format", cfg.tts.default_format)).lower()
        if self.fmt not in cfg.tts.supported_formats:
            raise TaskError("UnsupportedFeature", f"format '{self.fmt}' is not supported")
        self.rate = int(params.get("sample_rate", cfg.tts.default_sample_rate))
        if self.rate not in cfg.tts.supported_sample_rates:
            raise TaskError("InvalidParameter", f"unsupported sample_rate: {self.rate}")
        volume = int(params.get("volume", 50))
        self.gain = max(0.0, min(2.0, volume / 50.0))

        self.buffer = TTSSentenceBuffer()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.cancel_event = threading.Event()
        self.worker: asyncio.Task | None = None
        self.error: BaseException | None = None
        self.total_chars = 0
        self.index = 0
        self.started = time.monotonic()
        self.first_audio_at: float | None = None
        self.last_continue = time.monotonic()

    async def start(self) -> None:
        self.worker = asyncio.create_task(self._run(), name=f"tts-{self.task_id}")

    # ----------------------------------------------------------- client input
    async def on_continue(self, payload: dict[str, Any]) -> None:
        if time.monotonic() - self.last_continue > MAX_CONTINUE_GAP_S:
            raise TaskError("InvalidParameter", "continue-task gap exceeded 23 seconds")
        self.last_continue = time.monotonic()
        text = (payload.get("input") or {}).get("text", "")
        if not isinstance(text, str) or not text:
            return
        if len(text) > MAX_SINGLE_CHUNK:
            raise TaskError("InvalidParameter", f"single continue-task exceeds {MAX_SINGLE_CHUNK} characters")
        self.total_chars += len(text)
        if self.total_chars > MAX_TOTAL_CHARS:
            raise TaskError("InvalidParameter", f"task exceeds {MAX_TOTAL_CHARS} characters")
        if self.error is not None:
            raise TaskError("InternalError", f"TTS worker failed: {self.error!r}")
        for sentence in self.buffer.feed(text):
            await self.queue.put(sentence)

    async def on_finish(self, payload: dict[str, Any]) -> dict:
        directive = (payload.get("input") or {}).get("directive")
        if directive == "cancel":
            await self.cancel()
            return {"characters": self.total_chars}
        if self.error is not None:
            raise TaskError("InternalError", f"TTS worker failed: {self.error!r}")
        for sentence in self.buffer.flush():
            await self.queue.put(sentence)
        await self.queue.put(None)
        if self.worker is not None:
            await self.worker
        if self.error is not None:
            raise TaskError("InternalError", f"TTS worker failed: {self.error!r}")
        return {"characters": self.total_chars}

    async def cancel(self) -> None:
        self.cancel_event.set()
        for sentence in self.buffer.flush():
            await self.queue.put(sentence)
        await self.queue.put(None)
        if self.worker is not None:
            try:
                await asyncio.wait_for(self.worker, timeout=10)
            except asyncio.TimeoutError:
                self.worker.cancel()

    # --------------------------------------------------------------- worker
    async def _run(self) -> None:
        try:
            while True:
                sentence = await self.queue.get()
                if sentence is None:
                    break
                if self.cancel_event.is_set():
                    continue
                await self.writer.send_json(
                    tts_result(self.task_id, "sentence-begin", self.index, sentence)
                )
                await self._synthesize_sentence(sentence)
                if not self.cancel_event.is_set():
                    await self.writer.send_json(
                        tts_result(
                            self.task_id,
                            "sentence-end",
                            self.index,
                            sentence,
                            words=[],
                            usage={"characters": self.total_chars},
                        )
                    )
                self.index += 1
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - reported at finish
            self.error = exc
            log.warning("TTS task %s failed: %r", self.task_id, exc)

    async def _synthesize_sentence(self, sentence: str) -> None:
        generator = self.backend.synthesize(
            sentence,
            self.voice,
            self.cancel_event,
            priority=0 if self.index == 0 else 1,
        )
        streamable = is_streamable(self.fmt)
        collected: list[np.ndarray] = []
        t0 = time.monotonic()
        n_samples = 0
        async for chunk in generator:
            if self.cancel_event.is_set():
                continue
            n_samples += int(len(chunk))
            if self.first_audio_at is None:
                self.first_audio_at = time.monotonic()
                METRICS.tts_first_chunk.observe(self.first_audio_at - self.started)
                METRICS.tts_characters.inc(len(sentence))
            if streamable:
                await self._send_chunk(chunk, sentence)
            else:
                collected.append(chunk)
        if not streamable and collected:
            await self._send_chunk(np.concatenate(collected), sentence)
        if n_samples:
            METRICS.tts_rtf.observe(max(0.0, time.monotonic() - t0) / (n_samples / 24000.0))

    async def _send_chunk(self, chunk: np.ndarray, sentence: str) -> None:
        try:
            data = encode_audio(chunk, 24000, self.fmt, self.rate, self.gain)
        except AudioFormatError as exc:
            raise TaskError("UnsupportedFeature", str(exc)) from exc
        await self.writer.send_json(
            tts_result(self.task_id, "sentence-synthesis", self.index, sentence, words=[])
        )
        await self.writer.send_bytes(data)
