from __future__ import annotations

import asyncio

from ..core.audio import pcm_duration_ms
from ..core.vad import MockVadStream, VadStream
from ..schemas import Final, Partial, StreamParams, Transcript, TranscriptSegment, Word
from ..settings import MockEngineConfig, ModelConfig
from .base import ASREngine, ASRStream


def _words(text: str, begin_ms: int, end_ms: int) -> tuple[Word, ...]:
    if not text:
        return ()
    step = max(1, (end_ms - begin_ms) // len(text))
    result = []
    for index, char in enumerate(text):
        start = begin_ms + index * step
        finish = end_ms if index == len(text) - 1 else start + step
        result.append(
            Word(
                begin_time=start,
                end_time=finish,
                text=char,
                punctuation=char if char in "，。！？、,.!?" else "",
            )
        )
    return tuple(result)


class MockASRStream(ASRStream):
    def __init__(self, config: MockEngineConfig, params: StreamParams) -> None:
        self._config = config
        self._params = params
        self._pcm = bytearray()
        self._emitted_chars = 0
        self._last_partial_ms = 0

    async def feed(self, pcm: bytes) -> list[Partial]:
        if self._config.latency_ms:
            await asyncio.sleep(self._config.latency_ms / 1000.0)
        self._pcm.extend(pcm)
        duration = pcm_duration_ms(bytes(self._pcm))
        if duration - self._last_partial_ms < self._config.partial_interval_ms:
            return []
        if duration < 100:
            return []
        self._last_partial_ms = duration
        sentence = self._sentence()
        self._emitted_chars = min(len(sentence), self._emitted_chars + 1)
        text = sentence[: self._emitted_chars]
        if not text:
            return []
        return [
            Partial(
                text=text,
                begin_time=self._params.extra.get("begin_time_ms", 0),
                end_time=None,
            )
        ]

    async def finish(self) -> Final | None:
        duration = pcm_duration_ms(bytes(self._pcm))
        if duration <= 0:
            return None
        begin = int(self._params.extra.get("begin_time_ms", 0))
        text = self._sentence()
        words = _words(text, begin, begin + duration) if self._config.emit_words else ()
        return Final(text=text, begin_time=begin, end_time=begin + duration, words=words)

    def _sentence(self) -> str:
        sentences = self._config.sentences or [""]
        index = int(self._params.extra.get("segment_index", 0)) % len(sentences)
        return sentences[index]


class MockEngine(ASREngine):
    """Deterministic engine used by tests, local protocol work and CI.

    It never imports numpy/torch, so it runs on the base install.
    """

    def __init__(self, config: MockEngineConfig) -> None:
        self._config = config

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def ready(self) -> bool:
        return True

    async def warmup(self) -> None:
        return None

    def create_stream(self, model: ModelConfig, params: StreamParams) -> ASRStream:
        return MockASRStream(self._config, params)

    def create_vad_stream(
        self, model: ModelConfig, params: StreamParams
    ) -> VadStream:
        return MockVadStream(self._config.segment_ms)

    async def transcribe(
        self, model: ModelConfig, pcm: bytes, params: StreamParams
    ) -> Transcript:
        if self._config.latency_ms:
            await asyncio.sleep(self._config.latency_ms / 1000.0)
        duration_ms = pcm_duration_ms(pcm)
        sentences = self._config.sentences or [""]
        if not sentences:
            sentences = [""]
        segment_ms = max(1, duration_ms // len(sentences))
        segments = []
        for index, sentence in enumerate(sentences):
            begin = index * segment_ms
            end = duration_ms if index == len(sentences) - 1 else (index + 1) * segment_ms
            segments.append(
                TranscriptSegment(
                    text=sentence,
                    begin_time=begin,
                    end_time=end,
                    words=list(_words(sentence, begin, end))
                    if self._config.emit_words
                    else [],
                )
            )
        return Transcript(
            text="".join(sentences),
            language=params.language,
            duration_ms=duration_ms,
            segments=segments,
        )
