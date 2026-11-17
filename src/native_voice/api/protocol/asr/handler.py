"""Bailian-compatible ASR task: run-task -> PCM frames -> finish-task."""
from __future__ import annotations

import logging
import time
from typing import Any

from ....audio.pcm import pcm16_to_float32
from ....audio.resample import resample
from ....backends.base import ASRBackend
from ....config.schema import ServiceConfig
from ....observability.metrics import METRICS
from ..bailian.errors import TaskError
from ..bailian.events import asr_result
from ..bailian.writer import ConnectionWriter
from .segmenter import SentenceSegmenter

log = logging.getLogger(__name__)


class ASRTaskHandler:
    def __init__(
        self,
        task_id: str,
        params: dict[str, Any],
        backend: ASRBackend,
        writer: ConnectionWriter,
        cfg: ServiceConfig,
    ):
        self.task_id = task_id
        self.params = params
        self.backend = backend
        self.writer = writer
        self.cfg = cfg
        self.fmt = str(params.get("format", "pcm")).lower()
        if self.fmt not in ("pcm", "wav"):
            raise TaskError("UnsupportedFeature", f"format '{self.fmt}' is not supported for streaming ASR")
        self.sample_rate = int(params.get("sample_rate", 16000))
        self.max_silence_ms = int(params.get("max_sentence_silence", cfg.asr.max_sentence_silence_ms))
        self.noise_threshold = float(params.get("speech_noise_threshold", 0.0))
        self.session = backend.create_session()
        self.segmenter = SentenceSegmenter(
            rate=16000,
            max_silence_ms=self.max_silence_ms,
            noise_threshold=self.noise_threshold,
        )
        self.sentence_id = 1
        self.active = False
        self.begin_ms = 0.0
        self.last_text = ""
        self.wav_header_seen = False
        self.started_at = time.monotonic()
        self.last_partial_at = self.started_at

    # ------------------------------------------------------------- lifecycle
    async def on_audio(self, data: bytes) -> None:
        if self.fmt == "wav" and not self.wav_header_seen:
            self.wav_header_seen = True
            if data[:4] == b"RIFF":
                data = data[44:]
        audio = pcm16_to_float32(data)
        if audio.size == 0:
            return
        if self.sample_rate != 16000:
            audio = resample(audio, self.sample_rate, 16000)
        events = self.segmenter.push(audio)
        try:
            text = await self.session.feed(audio)
        except Exception as exc:
            raise TaskError("InternalError", f"ASR inference failed: {exc!r}") from exc

        for ev in events:
            if ev.kind == "start":
                await self._emit_partial(text="", sentence_begin=True, begin_ms=ev.begin_ms)
                self.active = True
                self.begin_ms = ev.begin_ms
                self.last_text = ""
            elif ev.kind == "end" and self.active:
                await self._finish_sentence(ev.end_ms, fallback=text)

        if self.active and text and text != self.last_text:
            await self._emit_partial(
                text=text,
                sentence_begin=False,
                begin_ms=self.begin_ms,
                end_ms=None,
                sentence_end=False,
            )
            now = time.monotonic()
            METRICS.asr_partial.observe(max(0.0, now - self.last_partial_at))
            self.last_partial_at = now
            self.last_text = text

    async def on_finish(self, payload: dict[str, Any]) -> dict:
        if self.active:
            await self._finish_sentence(self.segmenter.last_voiced_ms, fallback=self.last_text)
        await self.session.close()
        usage = {"duration": max(0, int(round(self.segmenter.audio_seconds)))}
        METRICS.tasks_total.labels("asr", "success").inc()
        METRICS.audio_seconds.labels("asr").inc(self.segmenter.audio_seconds)
        return usage

    async def cancel(self) -> None:
        self.active = False
        try:
            await self.session.close()
        except Exception:
            pass

    # --------------------------------------------------------------- helpers
    async def _finish_sentence(self, end_ms: float, fallback: str) -> None:
        try:
            final = await self.session.finalize()
        except Exception as exc:
            raise TaskError("InternalError", f"ASR finalize failed: {exc!r}") from exc
        final = final or fallback or self.last_text
        if final:
            sentence = {
                "begin_time": int(self.begin_ms),
                "end_time": int(end_ms),
                "text": final,
                "heartbeat": False,
                "sentence_begin": False,
                "sentence_end": True,
                "sentence_id": self.sentence_id,
                "words": [],
            }
            usage = {"duration": max(0, int(round(self.segmenter.audio_seconds)))}
            await self.writer.send_json(asr_result(self.task_id, sentence, usage=usage))
        self.active = False
        self.last_text = ""
        self.sentence_id += 1
        await self.session.reset()

    async def _emit_partial(
        self,
        text: str,
        sentence_begin: bool,
        begin_ms: float,
        end_ms: float | None = None,
        sentence_end: bool = False,
    ) -> None:
        sentence = {
            "begin_time": int(begin_ms),
            "end_time": int(end_ms) if end_ms is not None else None,
            "text": text,
            "heartbeat": False,
            "sentence_begin": sentence_begin,
            "sentence_end": sentence_end,
            "sentence_id": self.sentence_id,
            "words": [],
        }
        await self.writer.send_json(asr_result(self.task_id, sentence, usage=None))
